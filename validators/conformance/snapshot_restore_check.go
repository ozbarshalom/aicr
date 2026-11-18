// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"time"

	"aicr.run/pkg/defaults"
	"aicr.run/pkg/errors"
	"aicr.run/pkg/validator/labels"
	validatorv1 "aicr.run/pkg/validator/v1"
	"aicr.run/validators"
	"aicr.run/validators/internal/allocmode"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
)

// snapshot-restore proves that Snapshot (ai-dynamo/snapshot, GPU checkpoint and
// restore) works on the cluster end to end, which its prerequisites alone do
// not show. It runs a tiny CUDA probe under a SnapshotJob, which checkpoints it,
// restores the checkpoint into a new pod, and checks that what the probe held
// came back: a token in the process's memory and the same token in GPU memory.
//
// What the probe is. A Python process in a stock python image, which holds no
// CUDA toolkit: it opens libcuda.so.1 (the container runtime injects it, as
// driver-runtime-management relies on), allocates 1 MiB of device memory and
// fills it with a pattern built from a random token. It then raises Snapshot's
// ready-for-snapshot sentinel, waits for restore-complete, reads the device
// memory back and writes "match <token>" or "mismatch <token>" to
// /snapshot-control/probe-result. That is Snapshot's workload contract
// (docs/reference/workload-contract.md): quiesce, signal, wait for the agent.
//
// How a restore is told from a fresh start. The source pod holds token A and
// the restore pod's own environment holds a different token B. The restore pod
// runs an inert command, so only the restored process can write probe-result,
// and it can only say "match A": A lives in the checkpointed memory, never in
// the restore pod's environment. The restore pod's readiness probe passes only
// when probe-result equals "match A", so a Ready restore pod is the verdict.
//
// Placement. The probe is allocated through the device plugin (nvidia.com/gpu),
// on a device-plugin node, and the restore pod is pinned to the node the
// checkpoint was taken on. A restore onto another node is a separate
// capability (it needs a shared store) and is not what this check certifies.
//
// Cleanup. Every namespaced object lives in a per-run namespace whose cleanup is
// registered before the first Create. Deleting the namespace deletes the
// PodSnapshot, and the Snapshot operator removes the cluster-scoped
// PodSnapshotContent and the stored checkpoint after it.
//
// Probe image. The default is a stock python image, fully qualified like the
// driver-runtime-management probe. It is NOT rewritten by
// AICR_VALIDATOR_IMAGE_REGISTRY, which names where the validator images live;
// AICR_SNAPSHOT_PROBE_IMAGE carries a full replacement reference for mirrors.
// The image needs python3, sh, cat and sleep. The orchestrator forwards the
// variable (pkg/validator/v1 defines the same literal; keep in sync).
const (
	snapshotComponentName = "snapshot"
	// snapshotDefaultNamespace is where the snapshot component installs when the
	// recipe's componentRef names no namespace.
	snapshotDefaultNamespace = "snapshot"
	snapshotOperatorName     = "snapshot-operator"
	snapshotAgentName        = "snapshot-agent"

	snapshotSourcePrefix  = "snapshot-probe-"
	snapshotRestorePrefix = "snapshot-restore-"
	snapshotProbeImage    = "docker.io/library/python:3.12-slim"
	snapshotProbeImageEnv = "AICR_SNAPSHOT_PROBE_IMAGE"

	// snapshotTargetContainer is the container the SnapshotJob checkpoints and
	// the restore pod restores into. "main" is the SnapshotJob default.
	snapshotTargetContainer = "main"

	// nvidiaRuntimeClass is set on the probe pods when the cluster has it,
	// which the GPU Operator creates. Without it the pods use the node's
	// default runtime.
	nvidiaRuntimeClass = "nvidia"

	snapshotControlVolume = "snapshot-control"
	snapshotControlDir    = "/snapshot-control"
	snapshotControlEnv    = "SNAPSHOT_CONTROL_DIR"
	snapshotProbeResult   = snapshotControlDir + "/probe-result"

	// snapshotSeccompProfile is the localhost profile the Snapshot chart installs
	// on each node. It blocks io_uring, which CRIU cannot checkpoint.
	snapshotSeccompProfile = "profiles/block-iouring.json"

	snapshotRestoreFromAnnotation = "nvidia.com/restore-from"
	snapshotRestoredCondition     = "nvidia.com/Restored"
	snapshotProbeTokenEnv         = "AICR_PROBE_TOKEN" //nolint:gosec // G101: env var name, not a credential
	snapshotProbeExpectEnv        = "AICR_PROBE_EXPECT"

	// Reasons of the nvidia.com/Restored pod condition that end a restore without
	// success, from Snapshot's api/podcontract (v0.3.0). "RestoreInProgress" and
	// "RestoreSucceeded" are the other two; the check never reads them, because
	// the readiness gate is the verdict.
	snapshotRestoreFailedReason   = "RestoreFailed"
	snapshotRestoreIncompatReason = "RestoreIncompatible"
	snapshotRestorePartialReason  = "RestorePartiallySucceeded"

	// Condition types and status of the SnapshotJob and PodSnapshot.
	snapshotConditionTrue  = "True"
	snapshotJobCaptured    = "Captured"
	snapshotJobCompleted   = "Completed"
	snapshotJobFailed      = "Failed"
	snapshotPodSnapshotRdy = "Ready"

	// snapshotProbeTerminationGrace lets the probe pods go at once: they hold
	// nothing worth draining, and the namespace cleanup has 30 seconds.
	snapshotProbeTerminationGrace = int64(1)

	// snapshotRestoreWorkBudget is the fallback work budget for a 10m check
	// (see driverRuntimeWorkBudget); a live check derives it from its deadline.
	snapshotRestoreWorkBudget = 8*time.Minute + 30*time.Second

	snapshotLogTailLines = int64(20)

	// noContainerState is podContainerStateSummary's answer for a pod none of
	// whose containers has terminated.
	noContainerState = "no container state reported"
)

var (
	snapshotJobGVR        = schema.GroupVersionResource{Group: apiGroupNVIDIA, Version: versionV1alpha1, Resource: "snapshotjobs"}
	podSnapshotGVR        = schema.GroupVersionResource{Group: apiGroupNVIDIA, Version: versionV1alpha1, Resource: "podsnapshots"}
	podSnapshotContentGVR = schema.GroupVersionResource{Group: apiGroupNVIDIA, Version: versionV1alpha1, Resource: "podsnapshotcontents"}

	// The waits. A capture includes pulling the probe image on a node that lacks
	// it. Both fit inside snapshotRestoreWorkBudget with the namespace cleanup.
	// They are variables so tests can shorten them.
	snapshotCaptureTimeout = 4 * time.Minute
	snapshotRestoreTimeout = 3 * time.Minute
	snapshotPollInterval   = defaults.PodPollInterval

	// snapshotProbeResultReader reads probe-result from the restored container;
	// tests replace it, since a fake clientset has no exec endpoint.
	snapshotProbeResultReader = execSnapshotProbeResult
)

// snapshotProbeScript is the probe described above. The agent mounts the
// control directory and sets SNAPSHOT_CONTROL_DIR; the Python is deliberately
// plain ctypes so the image needs nothing but python3.
const snapshotProbeScript = `import ctypes
import os
import pathlib
import sys
import time

ctl = pathlib.Path(os.environ.get("SNAPSHOT_CONTROL_DIR", "/snapshot-control"))
(ctl / "ready-for-snapshot").unlink(missing_ok=True)


def fail(message):
    print("probe error: " + message, flush=True)
    sys.exit(1)


try:
    cuda = ctypes.CDLL("libcuda.so.1")
except OSError as exc:
    fail("libcuda.so.1 not found, so the container runtime injected no NVIDIA driver: %s" % exc)


def check(rc, what):
    if rc != 0:
        fail("%s failed with CUresult %d" % (what, rc))


check(cuda.cuInit(0), "cuInit")
dev = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
size = 1 << 20
dptr = ctypes.c_uint64()
check(cuda.cuMemAlloc_v2(ctypes.byref(dptr), ctypes.c_size_t(size)), "cuMemAlloc")
token = os.environ["AICR_PROBE_TOKEN"]
raw = token.encode()
pattern = (raw * (size // len(raw) + 1))[:size]
check(cuda.cuMemcpyHtoD_v2(dptr, pattern, ctypes.c_size_t(size)), "cuMemcpyHtoD")
check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
print("probe ready token=" + token, flush=True)
(ctl / "ready-for-snapshot").write_text("ready\n")

while not (ctl / "restore-complete").exists():
    time.sleep(0.2)

buf = ctypes.create_string_buffer(size)
check(cuda.cuMemcpyDtoH_v2(buf, dptr, ctypes.c_size_t(size)), "cuMemcpyDtoH")
verdict = "match" if buf.raw == pattern else "mismatch"
(ctl / "probe-result").write_text("%s %s\n" % (verdict, token))
while True:
    time.sleep(3600)
`

// snapshotRestoreRun carries the names and tokens of one run.
type snapshotRestoreRun struct {
	run       *gpuTestRun
	jobName   string
	podName   string
	srcToken  string
	restToken string
	image     string
	// runtimeClass is "" when the cluster has no nvidia RuntimeClass.
	runtimeClass string
}

// expected is what the restored process writes to probe-result: the verdict and
// the token the source pod held.
func (r *snapshotRestoreRun) expected() string { return "match " + r.srcToken }

// CheckSnapshotRestore checkpoints a CUDA probe with Snapshot, restores it, and
// requires the restored process to report that its memory and GPU memory
// survived. It skips unless the recipe declares the snapshot component, and
// once it is declared a prerequisite that is missing fails the check.
func CheckSnapshotRestore(ctx *validators.Context) error {
	if ctx.Clientset == nil {
		return errors.New(errors.ErrCodeInvalidRequest, "kubernetes client is not available")
	}
	if !validators.RecipeDeclares(ctx, snapshotComponentName) {
		return validators.SkipWithReason(validators.SkipReasonComponentNotInRecipe,
			"snapshot is not declared in the recipe, so there is no Snapshot checkpoint and restore to verify")
	}

	budget, err := gpuCheckWorkBudget(ctx.Ctx, snapshotRestoreWorkBudget)
	if err != nil {
		return err
	}
	workCtx, cancelWork := ctx.Timeout(budget)
	defer cancelWork()
	work := *ctx
	work.Ctx = workCtx
	ctx = &work

	dynClient, err := getDynamicClient(ctx)
	if err != nil {
		return err
	}
	namespace := snapshotComponentNamespace(ctx)
	if err = verifySnapshotInstalled(ctx, namespace); err != nil {
		return err
	}

	mode, err := detectGPUAllocationMode(ctx.Ctx, ctx.Clientset, dynClient)
	if err != nil {
		return err
	}
	recordRawTextArtifact(ctx, "GPU allocation mode",
		"kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu",
		mode.Summary())
	policy := ctx.ValidationInput.GetGPUAllocationPolicy()
	validators.EmitExtraOrWarn(validators.GPUAllocationPolicyExtra(policy))
	if verifyErr := verifyGPUAllocationPolicy(policy, mode); verifyErr != nil {
		return verifyErr
	}
	if err = requireDevicePluginProbe(policy, mode); err != nil {
		return err
	}
	validators.EmitExtraOrWarn(validators.GPUAllocationResolvedExtra(false, false))

	image := resolveSnapshotProbeImage()
	recordRawTextArtifact(ctx, "Snapshot probe image", "",
		fmt.Sprintf("image: %s (override: %s)", image, snapshotProbeImageOverrideNote()))
	return runSnapshotRestoreTest(ctx, dynClient, mode.DevicePluginNodes, image)
}

// requireDevicePluginProbe fails when the probe cannot be allocated through the
// device plugin: a recipe that pins DRA allocation, or a cluster without
// device-plugin GPUs. The probe is a plain pod inside a SnapshotJob, and
// Snapshot's own documentation covers device-plugin GPUs.
func requireDevicePluginProbe(policy string, mode *allocmode.Mode) error {
	if policy == validatorv1.GPUAllocationPolicyDRAResourceClaim {
		return errors.New(errors.ErrCodeUnavailable,
			"the Snapshot probe is allocated through the device plugin (nvidia.com/gpu), but the recipe's GPU allocation policy is "+policy)
	}
	if !mode.DevicePluginUsable || len(mode.DevicePluginNodes) == 0 {
		return errors.New(errors.ErrCodeUnavailable, fmt.Sprintf(
			"no device-plugin GPU node for the Snapshot probe: %s", mode.DevicePluginDetail))
	}
	return nil
}

// snapshotComponentNamespace returns the namespace the recipe installs snapshot
// into, or the default.
func snapshotComponentNamespace(ctx *validators.Context) string {
	if ctx.ValidationInput != nil {
		for _, ref := range ctx.ValidationInput.ComponentRefs {
			if ref.Name == snapshotComponentName && ref.Namespace != "" {
				return ref.Namespace
			}
		}
	}
	return snapshotDefaultNamespace
}

// verifySnapshotInstalled requires the operator and the node agent to be up
// before any GPU is requested, so a missing install reads as one.
func verifySnapshotInstalled(ctx *validators.Context, namespace string) error {
	if err := verifyDeploymentAvailable(ctx, namespace, snapshotOperatorName); err != nil {
		return errors.PropagateOrWrap(err, errors.ErrCodeUnavailable,
			fmt.Sprintf("snapshot operator %s/%s is not available", namespace, snapshotOperatorName))
	}
	if err := verifyDaemonSetReady(ctx, namespace, snapshotAgentName); err != nil {
		return errors.PropagateOrWrap(err, errors.ErrCodeUnavailable,
			fmt.Sprintf("snapshot node agent %s/%s is not ready", namespace, snapshotAgentName))
	}
	return nil
}

// resolveSnapshotProbeImage returns the AICR_SNAPSHOT_PROBE_IMAGE reference
// verbatim when set, else the default.
func resolveSnapshotProbeImage() string {
	if override := strings.TrimSpace(os.Getenv(snapshotProbeImageEnv)); override != "" {
		return override
	}
	return snapshotProbeImage
}

func snapshotProbeImageOverrideNote() string {
	if strings.TrimSpace(os.Getenv(snapshotProbeImageEnv)) != "" {
		return snapshotProbeImageEnv
	}
	return valueNone
}

// randomProbeToken returns 64 bits of entropy as hex.
func randomProbeToken() (string, error) {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", errors.Wrap(errors.ErrCodeInternal, "failed to generate a probe token", err)
	}
	return hex.EncodeToString(b), nil
}

// runSnapshotRestoreTest creates the per-run namespace, captures the probe,
// restores it and judges the result. The return is NAMED so the deferred
// cleanup can fail an otherwise-passing check when cleanup terminally fails; a
// primary error is never overwritten.
func runSnapshotRestoreTest(ctx *validators.Context, dynClient dynamic.Interface, nodes []string, image string) (err error) {
	run, runErr := newGPUTestRun()
	if runErr != nil {
		return runErr
	}
	srcToken, err := randomProbeToken()
	if err != nil {
		return err
	}
	restToken, err := randomProbeToken()
	if err != nil {
		return err
	}
	r := &snapshotRestoreRun{
		run:       run,
		jobName:   snapshotSourcePrefix + run.token,
		podName:   snapshotRestorePrefix + run.token,
		srcToken:  srcToken,
		restToken: restToken,
		image:     image,
	}

	// Register cleanup BEFORE the first Create: an ambiguously created probe
	// would otherwise stay bound and hold a GPU (see cleanupGPUTestNamespace).
	defer func() { //nolint:contextcheck // cleanup runs on its own bounded context
		status, cleanupErr := cleanupGPUTestNamespace(ctx.Clientset, run)
		recordRawTextArtifact(ctx, "Cleanup test namespace", cleanupInspectEquivalent(run.namespace), status)
		if cleanupErr != nil && err == nil {
			err = cleanupErr
		}
	}()

	if err = createGPUTestNamespace(ctx.Ctx, ctx.Clientset, run); err != nil {
		return err
	}
	r.runtimeClass = snapshotRuntimeClass(ctx)

	// --- capture ---
	captureStart := time.Now()
	job, err := buildSnapshotJob(r, ctx.Tolerations, nodes)
	if err != nil {
		return err
	}
	if _, err = dynClient.Resource(snapshotJobGVR).Namespace(run.namespace).Create(ctx.Ctx, job, metav1.CreateOptions{}); err != nil {
		if k8serrors.IsNotFound(err) {
			return errors.Wrap(errors.ErrCodeUnavailable,
				"the SnapshotJob API (snapshotjobs.nvidia.com) is not served: is the snapshot component's CRD installed?", err)
		}
		return errors.Wrap(errors.ErrCodeInternal, "failed to create the Snapshot probe SnapshotJob", err)
	}
	completed, captureErr := waitForSnapshotCapture(ctx, dynClient, r)
	recordRawTextArtifact(ctx, "SnapshotJob",
		fmt.Sprintf("kubectl get snapshotjob %s -n %s -o yaml", r.jobName, run.namespace), describeSnapshotJob(completed))
	if captureErr != nil {
		return captureErr
	}
	snapshotName, _, _ := unstructured.NestedString(completed.Object, "status", "podSnapshotName")
	if snapshotName == "" {
		return errors.New(errors.ErrCodeInternal, fmt.Sprintf(
			"SnapshotJob %s completed without naming a PodSnapshot (status.podSnapshotName is empty)", r.jobName))
	}
	node, err := verifyPodSnapshot(ctx, dynClient, run.namespace, snapshotName)
	if err != nil {
		return err
	}
	captureTook := time.Since(captureStart).Round(time.Second)
	recordRawTextArtifact(ctx, "Snapshot capture", fmt.Sprintf("kubectl get podsnapshot %s -n %s", snapshotName, run.namespace),
		fmt.Sprintf("PodSnapshot: %s/%s\nCaptured on node: %s\nCapture took: %s", run.namespace, snapshotName, valueOrUnknown(node), captureTook))

	// --- restore ---
	restoreStart := time.Now()
	pod := buildSnapshotRestorePod(r, snapshotName, ctx.Tolerations, nodes, node)
	if err = createPodWhenSAReady(ctx.Ctx, ctx.Clientset, run.namespace, pod); err != nil {
		return errors.PropagateOrWrap(err, errors.ErrCodeInternal, "failed to create the Snapshot restore pod")
	}
	restored, restoreErr := waitForSnapshotRestore(ctx, r)
	restoreTook := time.Since(restoreStart).Round(time.Second)
	result, resultErr := snapshotProbeResultReader(ctx, r)
	recordRawTextArtifact(ctx, "Snapshot restore pod",
		fmt.Sprintf("kubectl get pod %s -n %s -o wide", r.podName, run.namespace), describeRestorePod(restored, result, resultErr))
	if restoreErr != nil {
		return restoreErr
	}
	reported := fmt.Sprintf("%q", result)
	if resultErr != nil {
		reported = "not read (" + resultErr.Error() + "); the pod's readiness probe passed only on " + fmt.Sprintf("%q", r.expected())
	}
	recordRawTextArtifact(ctx, "Snapshot restore verdict", "", fmt.Sprintf(
		"source token (in the checkpointed process and GPU memory): %s\nrestore pod environment token (not restored):             %s\nrestored process reported: %s\nrestore took: %s\nverdict: PASS",
		r.srcToken, r.restToken, reported, restoreTook))
	return nil
}

// snapshotRuntimeClass returns the nvidia RuntimeClass name when the cluster
// has it, else "".
func snapshotRuntimeClass(ctx *validators.Context) string {
	if _, err := ctx.Clientset.NodeV1().RuntimeClasses().Get(ctx.Ctx, nvidiaRuntimeClass, metav1.GetOptions{}); err == nil {
		return nvidiaRuntimeClass
	}
	return ""
}

// snapshotTolerations returns the pod tolerations: the context's when set,
// else tolerate-all, as the other GPU probes do.
func snapshotTolerations(tolerations []corev1.Toleration) []corev1.Toleration {
	if tolerations != nil {
		return tolerations
	}
	return []corev1.Toleration{{Operator: corev1.TolerationOpExists}}
}

// buildSnapshotJob returns the SnapshotJob that runs and checkpoints the probe.
// The controller injects the control volume, the readiness gate and the
// seccomp profile; the image, command and GPU request are the caller's.
func buildSnapshotJob(r *snapshotRestoreRun, tolerations []corev1.Toleration, nodes []string) (*unstructured.Unstructured, error) {
	spec := corev1.PodSpec{
		RestartPolicy:                 corev1.RestartPolicyNever,
		TerminationGracePeriodSeconds: int64Ptr(snapshotProbeTerminationGrace),
		RuntimeClassName:              runtimeClassPtr(r.runtimeClass),
		Tolerations:                   snapshotTolerations(tolerations),
		Affinity:                      nodeNameAffinity(nodes),
		ImagePullSecrets:              r.run.imagePullSecretRefs(),
		Containers: []corev1.Container{{
			Name:    snapshotTargetContainer,
			Image:   r.image,
			Command: []string{"python3", "-c", snapshotProbeScript},
			Env:     []corev1.EnvVar{{Name: snapshotProbeTokenEnv, Value: r.srcToken}},
			Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
				corev1.ResourceName(resourceNVIDIAGPU): resource.MustParse("1"),
			}},
		}},
	}
	specMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&spec)
	if err != nil {
		return nil, errors.Wrap(errors.ErrCodeInternal, "failed to convert the Snapshot probe pod spec", err)
	}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": snapshotJobGVR.Group + "/" + snapshotJobGVR.Version,
		"kind":       "SnapshotJob",
		"metadata": map[string]any{
			keyName:     r.jobName,
			"namespace": r.run.namespace,
			"labels":    map[string]any{labels.GPUTestRun: r.run.token},
		},
		keySpec: map[string]any{
			"activeDeadlineSeconds": int64(snapshotCaptureTimeout / time.Second),
			"podSnapshotTemplate": map[string]any{
				"targetContainers": []any{snapshotTargetContainer},
			},
			"podTemplate": map[string]any{keySpec: specMap},
		},
	}}, nil
}

// buildSnapshotRestorePod returns the pod that restores the checkpoint. Its
// command is inert: Snapshot restores the checkpointed process into the
// container, and a runtime without the restore proxy would otherwise start the
// probe from scratch beside it. The pod is pinned to the node the checkpoint
// was taken on, or to the device-plugin nodes when that node is unknown.
func buildSnapshotRestorePod(r *snapshotRestoreRun, snapshotName string, tolerations []corev1.Toleration, nodes []string, captureNode string) *corev1.Pod {
	if captureNode != "" {
		nodes = []string{captureNode}
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        r.podName,
			Namespace:   r.run.namespace,
			Annotations: map[string]string{snapshotRestoreFromAnnotation: snapshotName},
		},
		Spec: corev1.PodSpec{
			RestartPolicy:                 corev1.RestartPolicyNever,
			TerminationGracePeriodSeconds: int64Ptr(snapshotProbeTerminationGrace),
			RuntimeClassName:              runtimeClassPtr(r.runtimeClass),
			Tolerations:                   snapshotTolerations(tolerations),
			Affinity:                      nodeNameAffinity(nodes),
			ImagePullSecrets:              r.run.imagePullSecretRefs(),
			Volumes: []corev1.Volume{{
				Name:         snapshotControlVolume,
				VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
			}},
			Containers: []corev1.Container{{
				Name:    snapshotTargetContainer,
				Image:   r.image,
				Command: []string{"sleep", "infinity"},
				Env: []corev1.EnvVar{
					{Name: snapshotControlEnv, Value: snapshotControlDir},
					// Not the source's token: a process that restarted instead of
					// restoring would report this one and fail the gate below.
					{Name: snapshotProbeTokenEnv, Value: r.restToken},
					{Name: snapshotProbeExpectEnv, Value: r.expected()},
				},
				SecurityContext: &corev1.SecurityContext{
					SeccompProfile: &corev1.SeccompProfile{
						Type:             corev1.SeccompProfileTypeLocalhost,
						LocalhostProfile: stringPtr(snapshotSeccompProfile),
					},
				},
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
					corev1.ResourceName(resourceNVIDIAGPU): resource.MustParse("1"),
				}},
				VolumeMounts: []corev1.VolumeMount{{
					Name: snapshotControlVolume, MountPath: snapshotControlDir, SubPath: snapshotTargetContainer,
				}},
				// The pod is Ready only when the restored process has written the
				// expected verdict. Nothing but the restored process can write it.
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{
						posixShell, "-c",
						fmt.Sprintf(`test "$(cat %s 2>/dev/null)" = "$%s"`, snapshotProbeResult, snapshotProbeExpectEnv),
					}}},
					PeriodSeconds: 1,
				},
			}},
		},
	}
}

func int64Ptr(v int64) *int64    { return &v }
func stringPtr(v string) *string { return &v }

func runtimeClassPtr(name string) *string {
	if name == "" {
		return nil
	}
	return &name
}

// snapshotCondition is one entry of a status.conditions list.
type snapshotCondition struct {
	Status  string
	Reason  string
	Message string
}

// snapshotConditions indexes obj's status.conditions by type.
func snapshotConditions(obj *unstructured.Unstructured) map[string]snapshotCondition {
	out := map[string]snapshotCondition{}
	if obj == nil {
		return out
	}
	items, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		t, _ := m["type"].(string)
		out[t] = snapshotCondition{
			Status:  stringField(m, "status"),
			Reason:  stringField(m, "reason"),
			Message: stringField(m, "message"),
		}
	}
	return out
}

func stringField(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

// describeSnapshotJob renders the SnapshotJob's conditions for evidence.
func describeSnapshotJob(obj *unstructured.Unstructured) string {
	if obj == nil {
		return "SnapshotJob was never read"
	}
	conds := snapshotConditions(obj)
	var b strings.Builder
	fmt.Fprintf(&b, "Name: %s/%s\n", obj.GetNamespace(), obj.GetName())
	for _, t := range []string{"Running", snapshotJobCaptured, snapshotJobCompleted, snapshotJobFailed} {
		c, ok := conds[t]
		if !ok {
			continue
		}
		fmt.Fprintf(&b, "%-10s %s (%s) %s\n", t+":", c.Status, c.Reason, c.Message)
	}
	return strings.TrimRight(b.String(), "\n")
}

// waitForSnapshotCapture polls the SnapshotJob until it is Completed, Failed or
// the capture timeout elapses. It returns the last SnapshotJob read. While the
// job has not captured, a source pod that failed or cannot start ends the wait
// at once with the pod's own explanation.
func waitForSnapshotCapture(ctx *validators.Context, dynClient dynamic.Interface, r *snapshotRestoreRun) (*unstructured.Unstructured, error) {
	waitCtx, cancel := context.WithTimeout(ctx.Ctx, snapshotCaptureTimeout)
	defer cancel()

	var last *unstructured.Unstructured
	var lastReadErr error
	err := wait.PollUntilContextCancel(waitCtx, snapshotPollInterval, true, func(c context.Context) (bool, error) {
		obj, getErr := dynClient.Resource(snapshotJobGVR).Namespace(r.run.namespace).Get(c, r.jobName, metav1.GetOptions{})
		if getErr != nil {
			if k8serrors.IsNotFound(getErr) {
				lastReadErr = nil
				return false, nil
			}
			if isK8sTimeoutErr(getErr) {
				if !readFailedBecauseContextEnded(c, getErr) {
					lastReadErr = getErr
				}
				return false, nil
			}
			return false, classifyK8sReadError(getErr, "SnapshotJob "+r.jobName)
		}
		lastReadErr = nil
		last = obj
		conds := snapshotConditions(obj)
		if f, ok := conds[snapshotJobFailed]; ok && f.Status == snapshotConditionTrue {
			return false, errors.New(errors.ErrCodeInternal, fmt.Sprintf(
				"SnapshotJob %s failed (%s): %s%s", r.jobName, f.Reason, f.Message, snapshotSourcePodState(c, ctx, r)))
		}
		if cm, ok := conds[snapshotJobCompleted]; ok && cm.Status == snapshotConditionTrue {
			return true, nil
		}
		if captured, ok := conds[snapshotJobCaptured]; ok && captured.Status == snapshotConditionTrue {
			return false, nil // captured; Completed follows once the source Job ends
		}
		return false, snapshotSourcePodFailure(c, ctx, r)
	})
	if err == nil {
		return last, nil
	}
	if ctx.Ctx.Err() != nil {
		return last, errors.Wrap(errors.ErrCodeTimeout, "waiting for the Snapshot capture canceled", ctx.Ctx.Err())
	}
	if waitCtx.Err() != nil {
		detail := "the SnapshotJob was never visible"
		if last != nil {
			detail = describeSnapshotJob(last)
		}
		if lastReadErr != nil {
			return last, errors.Wrap(errors.ErrCodeTimeout, fmt.Sprintf(
				"SnapshotJob %s unreadable for %s (reads kept failing)", r.jobName, snapshotCaptureTimeout), lastReadErr)
		}
		return last, errors.New(errors.ErrCodeTimeout, fmt.Sprintf(
			"SnapshotJob %s did not complete in %s (the source pod may be unschedulable or still pulling %s): %s%s",
			r.jobName, snapshotCaptureTimeout, r.image, detail, snapshotSourcePodState(ctx.Ctx, ctx, r)))
	}
	return last, errors.PropagateOrWrap(err, errors.ErrCodeInternal, "failed waiting for the Snapshot capture")
}

// snapshotSourcePod returns the pod the SnapshotJob's batch Job created, or nil.
func snapshotSourcePod(c context.Context, ctx *validators.Context, r *snapshotRestoreRun) *corev1.Pod {
	pods, err := ctx.Clientset.CoreV1().Pods(r.run.namespace).List(c, metav1.ListOptions{LabelSelector: "job-name=" + r.jobName})
	if err != nil || len(pods.Items) == 0 {
		return nil
	}
	// The batch Job may have replaced a failed pod: the newest one is the one
	// that matters.
	newest := &pods.Items[0]
	for i := range pods.Items {
		if pods.Items[i].CreationTimestamp.After(newest.CreationTimestamp.Time) {
			newest = &pods.Items[i]
		}
	}
	return newest
}

// snapshotSourcePodFailure returns an error when the source pod has failed or
// is stuck (an image that will not pull), and nil while it is still progressing.
func snapshotSourcePodFailure(c context.Context, ctx *validators.Context, r *snapshotRestoreRun) error {
	pod := snapshotSourcePod(c, ctx, r)
	if pod == nil {
		return nil
	}
	if reason := podStuckReason(pod); reason != "" {
		return podStuckErr("the Snapshot probe source pod "+pod.Name, reason)
	}
	if pod.Status.Phase == corev1.PodFailed {
		return errors.New(errors.ErrCodeInternal, fmt.Sprintf(
			"the Snapshot probe source pod %s failed before it was checkpointed: %s\nlogs: %s",
			pod.Name, podContainerStateSummary(pod), snapshotPodLogs(c, ctx, pod.Namespace, pod.Name)))
	}
	return nil
}

// snapshotSourcePodState describes the source pod for a timeout message.
func snapshotSourcePodState(parent context.Context, ctx *validators.Context, r *snapshotRestoreRun) string {
	c, cancel := context.WithTimeout(context.WithoutCancel(parent), defaults.K8sCleanupTimeout)
	defer cancel()
	pod := snapshotSourcePod(c, ctx, r)
	if pod == nil {
		return "\nsource pod: none created"
	}
	msg := fmt.Sprintf("\nsource pod %s: phase=%s, %s", pod.Name, pod.Status.Phase, podWaitingStatus(pod))
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodScheduled && cond.Status == corev1.ConditionFalse {
			msg += fmt.Sprintf("; not scheduled: %s", cond.Message)
		}
	}
	if ended := podContainerStateSummary(pod); ended != noContainerState {
		msg += "; " + ended
	}
	return msg
}

// podContainerStateSummary renders each container's terminated state.
func podContainerStateSummary(pod *corev1.Pod) string {
	var parts []string
	for _, cs := range pod.Status.ContainerStatuses {
		if t := cs.State.Terminated; t != nil {
			part := fmt.Sprintf("%s exited %d (%s)", cs.Name, t.ExitCode, t.Reason)
			if t.Message != "" {
				part += ": " + t.Message
			}
			parts = append(parts, part)
		}
	}
	if len(parts) == 0 {
		return noContainerState
	}
	return strings.Join(parts, "; ")
}

// snapshotPodLogs returns the tail of the probe container's logs, or the read
// error, for a failure message.
func snapshotPodLogs(c context.Context, ctx *validators.Context, namespace, name string) string {
	tail := snapshotLogTailLines
	raw, err := ctx.Clientset.CoreV1().Pods(namespace).GetLogs(name, &corev1.PodLogOptions{
		Container: snapshotTargetContainer, TailLines: &tail,
	}).DoRaw(c)
	if err != nil {
		return fmt.Sprintf("(could not read logs: %v)", err)
	}
	return strings.TrimSpace(string(raw))
}

// verifyPodSnapshot requires the PodSnapshot the SnapshotJob produced to be
// Ready, and returns the node the checkpoint was taken on ("" if unreadable).
func verifyPodSnapshot(ctx *validators.Context, dynClient dynamic.Interface, namespace, name string) (string, error) {
	snap, err := dynClient.Resource(podSnapshotGVR).Namespace(namespace).Get(ctx.Ctx, name, metav1.GetOptions{})
	if err != nil {
		return "", classifyK8sReadError(err, "PodSnapshot "+namespace+"/"+name)
	}
	ready, ok := snapshotConditions(snap)[snapshotPodSnapshotRdy]
	if !ok || ready.Status != snapshotConditionTrue {
		return "", errors.New(errors.ErrCodeInternal, fmt.Sprintf(
			"PodSnapshot %s/%s is not Ready after the SnapshotJob completed (%s %s)", namespace, name, ready.Reason, ready.Message))
	}
	contentName, _, _ := unstructured.NestedString(snap.Object, "status", "boundSnapshotContentName")
	if contentName == "" {
		return "", nil
	}
	content, err := dynClient.Resource(podSnapshotContentGVR).Get(ctx.Ctx, contentName, metav1.GetOptions{})
	if err != nil {
		return "", nil //nolint:nilerr // the node only narrows placement; the restore falls back to the device-plugin nodes
	}
	node, _, _ := unstructured.NestedString(content.Object, "spec", "source", "nodeName")
	return node, nil
}

// waitForSnapshotRestore polls the restore pod until it is Ready. It fails at
// once when the pod's Restored condition reports a failed restore or the pod
// ends, and on timeout it reports where the restore stood.
func waitForSnapshotRestore(ctx *validators.Context, r *snapshotRestoreRun) (*corev1.Pod, error) {
	waitCtx, cancel := context.WithTimeout(ctx.Ctx, snapshotRestoreTimeout)
	defer cancel()

	var last *corev1.Pod
	var lastReadErr error
	err := wait.PollUntilContextCancel(waitCtx, snapshotPollInterval, true, func(c context.Context) (bool, error) {
		pod, getErr := ctx.Clientset.CoreV1().Pods(r.run.namespace).Get(c, r.podName, metav1.GetOptions{})
		if getErr != nil {
			if k8serrors.IsNotFound(getErr) {
				lastReadErr = nil
				return false, nil
			}
			if isK8sTimeoutErr(getErr) {
				if !readFailedBecauseContextEnded(c, getErr) {
					lastReadErr = getErr
				}
				return false, nil
			}
			return false, classifyK8sReadError(getErr, "pod "+r.podName)
		}
		lastReadErr = nil
		last = pod
		if failure := snapshotRestoreFailure(pod); failure != "" {
			return false, errors.New(errors.ErrCodeInternal, fmt.Sprintf(
				"Snapshot restore into pod %s failed: %s", r.podName, failure))
		}
		return podIsReady(pod), nil
	})
	if err == nil {
		return last, nil
	}
	if ctx.Ctx.Err() != nil {
		return last, errors.Wrap(errors.ErrCodeTimeout, "waiting for the Snapshot restore canceled", ctx.Ctx.Err())
	}
	if waitCtx.Err() != nil {
		if lastReadErr != nil {
			return last, errors.Wrap(errors.ErrCodeTimeout, fmt.Sprintf(
				"restore pod %s unreadable for %s (reads kept failing)", r.podName, snapshotRestoreTimeout), lastReadErr)
		}
		return last, errors.New(errors.ErrCodeTimeout, fmt.Sprintf(
			"restore pod %s was not Ready in %s: the restored process did not report the expected result (%s)",
			r.podName, snapshotRestoreTimeout, describePodProgress(last)))
	}
	return last, err
}

// snapshotRestoreFailure returns why the restore cannot succeed, or "" while it
// may still: a failed Restored condition, a pod that has ended, or a pod stuck
// on something it will not recover from.
func snapshotRestoreFailure(pod *corev1.Pod) string {
	for _, cond := range pod.Status.Conditions {
		if string(cond.Type) != snapshotRestoredCondition || cond.Status == corev1.ConditionTrue {
			continue
		}
		switch cond.Reason {
		case snapshotRestoreFailedReason, snapshotRestoreIncompatReason, snapshotRestorePartialReason:
			return fmt.Sprintf("%s: %s", cond.Reason, cond.Message)
		}
	}
	if reason := podStuckReason(pod); reason != "" {
		return reason
	}
	if pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded {
		return "the pod ended (" + podContainerStateSummary(pod) + "); a restored pod's container ends when the restored process does, and the restore proxy exits 112 when the pod's one restore was already used and 113 when it lost the restored process"
	}
	return ""
}

// describePodProgress summarizes how far a restore pod got.
func describePodProgress(pod *corev1.Pod) string {
	if pod == nil {
		return "the pod was never visible"
	}
	msg := fmt.Sprintf("phase=%s, ready=%t", pod.Status.Phase, podIsReady(pod))
	if waiting := podWaitingStatus(pod); waiting != valueNone {
		msg += ", waiting " + waiting
	}
	for _, cond := range pod.Status.Conditions {
		if string(cond.Type) == snapshotRestoredCondition {
			msg += fmt.Sprintf("; Restored=%s (%s) %s", cond.Status, cond.Reason, cond.Message)
		}
	}
	return msg
}

// execSnapshotProbeResult reads probe-result from the restored container, for
// the evidence and for the failure message. The readiness gate, not this read,
// is the verdict; an exec that is not permitted never fails the check.
func execSnapshotProbeResult(ctx *validators.Context, r *snapshotRestoreRun) (string, error) {
	execCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx.Ctx), defaults.K8sCleanupTimeout)
	defer cancel()
	res, err := execPodCommand(execCtx, ctx, r.run.namespace, r.podName,
		[]string{"cat", snapshotProbeResult}, podExecOptions{PreferredContainerName: snapshotTargetContainer})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(res.Stdout), nil
}

// describeRestorePod renders the restore pod's outcome for evidence.
func describeRestorePod(pod *corev1.Pod, result string, resultErr error) string {
	if pod == nil {
		return "restore pod was never visible"
	}
	read := fmt.Sprintf("probe-result: %q", result)
	if resultErr != nil {
		read = fmt.Sprintf("probe-result: not read (%v)", resultErr)
	}
	return fmt.Sprintf("Name:   %s/%s\nNode:   %s\nStatus: %s\n%s",
		pod.Namespace, pod.Name, valueOrUnknown(pod.Spec.NodeName), describePodProgress(pod), read)
}
