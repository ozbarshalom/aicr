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
	stderrors "errors"
	"os/exec"
	"strings"
	"testing"
	"time"

	"aicr.run/pkg/errors"
	"aicr.run/pkg/errors/errorstest"
	"aicr.run/pkg/recipe"
	v1 "aicr.run/pkg/validator/v1"
	"aicr.run/validators"
	"aicr.run/validators/internal/allocmode"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

const (
	testSnapshotNamespace = "snapshot"
	testSnapshotNode      = "node1"
	testSnapshotContent   = "podsnapshotcontent-test"
)

// snapshotFake is a fake cluster with the Snapshot operator installed. It plays
// the parts of the operator and the node agent: a SnapshotJob is created with
// the status jobStatus gives it (a finished capture by default), together with
// the PodSnapshot and PodSnapshotContent a capture leaves behind, and the
// restore pod comes back with the status restoreStatus gives it (Ready and
// Restored by default).
type snapshotFake struct {
	clientset *k8sfake.Clientset
	dyn       *dynamicfake.FakeDynamicClient

	// jobs and pods are the objects as the check submitted them.
	jobs []*unstructured.Unstructured
	pods []*corev1.Pod

	// jobStatus sets the SnapshotJob's status when it is created. nil is a
	// finished capture.
	jobStatus func(job *unstructured.Unstructured)
	// sourcePod, when set, is added to the cluster as the pod the SnapshotJob's
	// batch Job created.
	sourcePod func(namespace, jobName string) *corev1.Pod
	// restoreStatus sets the restore pod's status when it is created. nil is a
	// Ready pod whose Restored condition is True.
	restoreStatus func(pod *corev1.Pod)
	// createErr, when set, fails the SnapshotJob create.
	createErr error
}

type snapshotFakeOpt func(*snapshotFake)

func snapshotConditionList(conds ...map[string]any) []any {
	out := make([]any, 0, len(conds))
	for _, c := range conds {
		out = append(out, c)
	}
	return out
}

func snapshotCond(t, status, reason, message string) map[string]any {
	return map[string]any{"type": t, "status": status, "reason": reason, "message": message}
}

// finishedCapture is the status of a SnapshotJob whose checkpoint is durable.
func finishedCapture(job *unstructured.Unstructured) {
	_ = unstructured.SetNestedSlice(job.Object, snapshotConditionList(
		snapshotCond("Running", "False", "JobCompleted", ""),
		snapshotCond("Captured", "True", "CaptureCompleted", ""),
		snapshotCond("Completed", "True", "JobCompleted", ""),
		snapshotCond("Failed", "False", "NoFailure", ""),
	), "status", "conditions")
	_ = unstructured.SetNestedField(job.Object, job.GetName(), "status", "podSnapshotName")
}

func readyRestoredPod(pod *corev1.Pod) {
	pod.Status.Phase = corev1.PodRunning
	pod.Status.Conditions = []corev1.PodCondition{
		{Type: corev1.PodReady, Status: corev1.ConditionTrue},
		{Type: corev1.PodConditionType(snapshotRestoredCondition), Status: corev1.ConditionTrue, Reason: "RestoreSucceeded"},
	}
}

func newSnapshotFake(t *testing.T, opts ...snapshotFakeOpt) *snapshotFake {
	t.Helper()
	// The probe image must come from the code under test, not from a developer
	// or CI shell that set the override.
	t.Setenv(snapshotProbeImageEnv, "")
	t.Setenv(validatorImageRegistryEnv, "")
	pollWas, captureWas, restoreWas, readerWas := snapshotPollInterval, snapshotCaptureTimeout, snapshotRestoreTimeout, snapshotProbeResultReader
	t.Cleanup(func() {
		snapshotPollInterval, snapshotCaptureTimeout, snapshotRestoreTimeout, snapshotProbeResultReader = pollWas, captureWas, restoreWas, readerWas
	})
	snapshotPollInterval = time.Millisecond
	snapshotCaptureTimeout = 2 * time.Second
	snapshotRestoreTimeout = 2 * time.Second
	snapshotProbeResultReader = func(_ *validators.Context, r *snapshotRestoreRun) (string, error) {
		return r.expected(), nil
	}

	f := &snapshotFake{jobStatus: finishedCapture, restoreStatus: readyRestoredPod}
	for _, opt := range opts {
		opt(f)
	}

	objs := []runtime.Object{
		testNode(testSnapshotNode, withGPUAllocatable("1")),
		&nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: nvidiaRuntimeClass}, Handler: nvidiaRuntimeClass},
		&appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: snapshotAgentName, Namespace: testSnapshotNamespace},
			Status:     appsv1.DaemonSetStatus{NumberReady: 1, DesiredNumberScheduled: 1},
		},
		&appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: snapshotOperatorName, Namespace: testSnapshotNamespace},
			Status:     appsv1.DeploymentStatus{AvailableReplicas: 1},
		},
	}
	f.clientset = k8sfake.NewClientset(objs...)
	withDRAAPIDiscovery(t, f.clientset)

	f.dyn = dynamicfake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(),
		map[schema.GroupVersionResource]string{
			draGVRAt("v1", "deviceclasses"):          "DeviceClassList",
			draGVRAt("v1", "resourceslices"):         "ResourceSliceList",
			draGVRAt("v1", "resourceclaims"):         "ResourceClaimList",
			draGVRAt("v1", "resourceclaimtemplates"): "ResourceClaimTemplateList",
			snapshotJobGVR:                           "SnapshotJobList",
			podSnapshotGVR:                           "PodSnapshotList",
			podSnapshotContentGVR:                    "PodSnapshotContentList",
		})

	f.dyn.PrependReactor("create", "snapshotjobs", func(action k8stesting.Action) (bool, runtime.Object, error) {
		create, ok := action.(k8stesting.CreateAction)
		if !ok {
			return false, nil, nil
		}
		job, ok := create.GetObject().(*unstructured.Unstructured)
		if !ok {
			return false, nil, nil
		}
		f.jobs = append(f.jobs, job.DeepCopy())
		if f.createErr != nil {
			return true, nil, f.createErr
		}
		f.jobStatus(job)
		if name, _, _ := unstructured.NestedString(job.Object, "status", "podSnapshotName"); name != "" {
			f.addCapture(t, job.GetNamespace(), name)
		}
		if f.sourcePod != nil {
			if err := f.clientset.Tracker().Add(f.sourcePod(job.GetNamespace(), job.GetName())); err != nil {
				t.Fatalf("add source pod: %v", err)
			}
		}
		return false, nil, nil // the tracker stores the SnapshotJob with its status
	})

	f.clientset.PrependReactor("create", "pods", func(action k8stesting.Action) (bool, runtime.Object, error) {
		create, ok := action.(k8stesting.CreateAction)
		if !ok {
			return false, nil, nil
		}
		pod, ok := create.GetObject().(*corev1.Pod)
		if !ok {
			return false, nil, nil
		}
		f.pods = append(f.pods, pod.DeepCopy())
		f.restoreStatus(pod)
		return false, nil, nil
	})
	return f
}

// addCapture stores the PodSnapshot and PodSnapshotContent a finished capture
// leaves behind.
func (f *snapshotFake) addCapture(t *testing.T, namespace, name string) {
	t.Helper()
	snap := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1alpha1",
		"kind":       "PodSnapshot",
		"metadata":   map[string]any{"name": name, "namespace": namespace},
		"status": map[string]any{
			"boundSnapshotContentName": testSnapshotContent,
			"conditions":               snapshotConditionList(snapshotCond("Ready", "True", "SnapshotReady", "")),
		},
	}}
	content := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1alpha1",
		"kind":       "PodSnapshotContent",
		"metadata":   map[string]any{"name": testSnapshotContent},
		"spec":       map[string]any{"source": map[string]any{"nodeName": testSnapshotNode}},
	}}
	if err := f.dyn.Tracker().Create(podSnapshotGVR, snap, namespace); err != nil {
		t.Fatalf("add PodSnapshot: %v", err)
	}
	if err := f.dyn.Tracker().Create(podSnapshotContentGVR, content, ""); err != nil {
		t.Fatalf("add PodSnapshotContent: %v", err)
	}
}

func (f *snapshotFake) context() *validators.Context {
	return &validators.Context{
		Ctx:           context.Background(),
		Clientset:     f.clientset,
		DynamicClient: f.dyn,
		ValidationInput: &v1.ValidationInput{ComponentRefs: []recipe.ComponentRef{
			{Name: snapshotComponentName, Namespace: testSnapshotNamespace},
		}},
	}
}

func (f *snapshotFake) run(t *testing.T) (string, error) {
	t.Helper()
	var err error
	out := captureStdout(t, func() { err = CheckSnapshotRestore(f.context()) })
	return out, err
}

func podSpecOf(t *testing.T, job *unstructured.Unstructured) corev1.PodSpec {
	t.Helper()
	raw, found, err := unstructured.NestedMap(job.Object, "spec", "podTemplate", "spec")
	if err != nil || !found {
		t.Fatalf("SnapshotJob has no spec.podTemplate.spec: found=%t err=%v", found, err)
	}
	var spec corev1.PodSpec
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(raw, &spec); err != nil {
		t.Fatalf("spec.podTemplate.spec does not decode as a PodSpec: %v", err)
	}
	return spec
}

func envValue(c corev1.Container, name string) string {
	for _, e := range c.Env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

func TestCheckSnapshotRestore_SkipsWhenSnapshotNotDeclared(t *testing.T) {
	f := newSnapshotFake(t)
	ctx := f.context()
	ctx.ValidationInput = &v1.ValidationInput{ComponentRefs: []recipe.ComponentRef{{Name: "gpu-operator"}}}

	err := CheckSnapshotRestore(ctx)
	if !validators.IsSkip(err) {
		t.Fatalf("CheckSnapshotRestore() = %v, want a skip", err)
	}
	if len(f.jobs) != 0 || len(f.pods) != 0 {
		t.Errorf("a skipped check created %d SnapshotJob(s) and %d pod(s), want none", len(f.jobs), len(f.pods))
	}
	requireNoLeakedNamespace(t, f.clientset)
}

func TestCheckSnapshotRestore_PassesAfterCaptureAndRestore(t *testing.T) {
	f := newSnapshotFake(t)
	out, err := f.run(t)
	if err != nil {
		t.Fatalf("CheckSnapshotRestore() error = %v\n%s", err, out)
	}

	if len(f.jobs) != 1 || len(f.pods) != 1 {
		t.Fatalf("created %d SnapshotJob(s) and %d pod(s), want 1 and 1", len(f.jobs), len(f.pods))
	}
	job, restore := f.jobs[0], f.pods[0]

	t.Run("SnapshotJob", func(t *testing.T) {
		if !strings.HasPrefix(job.GetName(), snapshotSourcePrefix) {
			t.Errorf("SnapshotJob name = %q, want prefix %q", job.GetName(), snapshotSourcePrefix)
		}
		targets, _, _ := unstructured.NestedStringSlice(job.Object, "spec", "podSnapshotTemplate", "targetContainers")
		if len(targets) != 1 || targets[0] != snapshotTargetContainer {
			t.Errorf("targetContainers = %v, want [%s]", targets, snapshotTargetContainer)
		}
		deadline, _, _ := unstructured.NestedInt64(job.Object, "spec", "activeDeadlineSeconds")
		if deadline != int64(snapshotCaptureTimeout/time.Second) {
			t.Errorf("activeDeadlineSeconds = %d, want the capture timeout", deadline)
		}
		spec := podSpecOf(t, job)
		if spec.RestartPolicy != corev1.RestartPolicyNever {
			t.Errorf("restartPolicy = %s, want Never", spec.RestartPolicy)
		}
		if spec.RuntimeClassName == nil || *spec.RuntimeClassName != nvidiaRuntimeClass {
			t.Errorf("runtimeClassName = %v, want %q when the cluster has it", spec.RuntimeClassName, nvidiaRuntimeClass)
		}
		if len(spec.Tolerations) != 1 || spec.Tolerations[0].Operator != corev1.TolerationOpExists {
			t.Errorf("tolerations = %v, want the tolerate-all default", spec.Tolerations)
		}
		if spec.Affinity == nil || spec.Affinity.NodeAffinity == nil {
			t.Error("source pod must be confined to the device-plugin nodes by node affinity")
		}
		if len(spec.Containers) != 1 {
			t.Fatalf("containers = %d, want 1", len(spec.Containers))
		}
		c := spec.Containers[0]
		if c.Name != snapshotTargetContainer || c.Image != snapshotProbeImage {
			t.Errorf("container = %s %s, want %s %s", c.Name, c.Image, snapshotTargetContainer, snapshotProbeImage)
		}
		if len(c.Command) != 3 || c.Command[0] != "python3" || c.Command[1] != "-c" || c.Command[2] != snapshotProbeScript {
			t.Errorf("command must run the probe script with python3 -c, got %v", c.Command)
		}
		if got := c.Resources.Limits[corev1.ResourceName(resourceNVIDIAGPU)]; got.Value() != 1 {
			t.Errorf("GPU limit = %s, want 1", got.String())
		}
		if len(envValue(c, snapshotProbeTokenEnv)) != 16 {
			t.Errorf("%s = %q, want 16 hex characters", snapshotProbeTokenEnv, envValue(c, snapshotProbeTokenEnv))
		}
	})

	t.Run("restore pod", func(t *testing.T) {
		if !strings.HasPrefix(restore.Name, snapshotRestorePrefix) {
			t.Errorf("restore pod name = %q, want prefix %q", restore.Name, snapshotRestorePrefix)
		}
		if got := restore.Annotations[snapshotRestoreFromAnnotation]; got != job.GetName() {
			t.Errorf("%s = %q, want the PodSnapshot the SnapshotJob produced (%q)", snapshotRestoreFromAnnotation, got, job.GetName())
		}
		if restore.Spec.RestartPolicy != corev1.RestartPolicyNever {
			t.Errorf("restartPolicy = %s, want Never (a pod gets one restore)", restore.Spec.RestartPolicy)
		}
		requireNodeNameAffinity(t, restore, []string{testSnapshotNode})
		c := restore.Spec.Containers[0]
		if c.Name != snapshotTargetContainer {
			t.Errorf("container name = %q, want %q (the checkpointed container's name)", c.Name, snapshotTargetContainer)
		}
		if strings.Join(c.Command, " ") != "sleep infinity" {
			t.Errorf("command = %v, want the inert sleep infinity: only the restored process may produce the result", c.Command)
		}
		if sc := c.SecurityContext; sc == nil || sc.SeccompProfile == nil || sc.SeccompProfile.Type != corev1.SeccompProfileTypeLocalhost ||
			sc.SeccompProfile.LocalhostProfile == nil || *sc.SeccompProfile.LocalhostProfile != snapshotSeccompProfile {

			t.Errorf("seccomp = %+v, want the Localhost profile %s", c.SecurityContext, snapshotSeccompProfile)
		}
		if got := envValue(c, snapshotControlEnv); got != snapshotControlDir {
			t.Errorf("%s = %q, want %q", snapshotControlEnv, got, snapshotControlDir)
		}
		if len(c.VolumeMounts) != 1 || c.VolumeMounts[0].MountPath != snapshotControlDir || c.VolumeMounts[0].SubPath != snapshotTargetContainer {
			t.Errorf("volumeMounts = %+v, want the control volume at %s with subPath %s", c.VolumeMounts, snapshotControlDir, snapshotTargetContainer)
		}
		if got := c.Resources.Limits[corev1.ResourceName(resourceNVIDIAGPU)]; got.Value() != 1 {
			t.Errorf("GPU limit = %s, want 1", got.String())
		}

		srcToken := envValue(probeContainerOf(t, job), snapshotProbeTokenEnv)
		restToken := envValue(c, snapshotProbeTokenEnv)
		if restToken == srcToken || restToken == "" {
			t.Errorf("restore pod token %q must differ from the source token %q, or a restart would pass as a restore", restToken, srcToken)
		}
		if got, want := envValue(c, snapshotProbeExpectEnv), "match "+srcToken; got != want {
			t.Errorf("%s = %q, want %q: the restore pod is Ready only on the SOURCE token", snapshotProbeExpectEnv, got, want)
		}
		if c.ReadinessProbe == nil || c.ReadinessProbe.Exec == nil {
			t.Fatal("restore pod needs an exec readiness probe: it is the verdict")
		}
		gate := strings.Join(c.ReadinessProbe.Exec.Command, " ")
		for _, want := range []string{snapshotProbeResult, "$" + snapshotProbeExpectEnv} {
			if !strings.Contains(gate, want) {
				t.Errorf("readiness gate %q lacks %q", gate, want)
			}
		}
	})

	for _, want := range []string{
		"--- SnapshotJob ---",
		"Completed: True",
		"--- Snapshot capture ---",
		"Captured on node: " + testSnapshotNode,
		"--- Snapshot restore verdict ---",
		"verdict: PASS",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout missing %q:\n%s", want, out)
		}
	}
	requireNoLeakedNamespace(t, f.clientset)
}

// probeContainerOf returns the probe container of a submitted SnapshotJob.
func probeContainerOf(t *testing.T, job *unstructured.Unstructured) corev1.Container {
	t.Helper()
	return podSpecOf(t, job).Containers[0]
}

func TestCheckSnapshotRestore_FailsWhenSnapshotNotInstalled(t *testing.T) {
	t.Run("operator missing", func(t *testing.T) {
		f := newSnapshotFake(t)
		if err := f.clientset.AppsV1().Deployments(testSnapshotNamespace).Delete(context.Background(), snapshotOperatorName, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("delete operator: %v", err)
		}
		_, err := f.run(t)
		if err == nil || !strings.Contains(err.Error(), snapshotOperatorName) {
			t.Fatalf("CheckSnapshotRestore() = %v, want an error naming %s", err, snapshotOperatorName)
		}
		if len(f.jobs) != 0 {
			t.Error("a missing operator must fail before any SnapshotJob is created")
		}
		requireNoLeakedNamespace(t, f.clientset)
	})
	t.Run("agent not ready", func(t *testing.T) {
		f := newSnapshotFake(t)
		ds, err := f.clientset.AppsV1().DaemonSets(testSnapshotNamespace).Get(context.Background(), snapshotAgentName, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("get agent: %v", err)
		}
		ds.Status.NumberReady = 0
		if _, err = f.clientset.AppsV1().DaemonSets(testSnapshotNamespace).Update(context.Background(), ds, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("update agent: %v", err)
		}
		_, err = f.run(t)
		if err == nil || !strings.Contains(err.Error(), snapshotAgentName) {
			t.Fatalf("CheckSnapshotRestore() = %v, want an error naming %s", err, snapshotAgentName)
		}
	})
}

func TestCheckSnapshotRestore_FailsWithoutSnapshotCRD(t *testing.T) {
	f := newSnapshotFake(t, func(f *snapshotFake) {
		f.createErr = k8serrors.NewNotFound(schema.GroupResource{Group: "nvidia.com", Resource: "snapshotjobs"}, "")
	})
	out, err := f.run(t)
	if err == nil {
		t.Fatalf("CheckSnapshotRestore() = nil, want an error\n%s", out)
	}
	errorstest.WantReportedCode(t, err, errors.ErrCodeUnavailable)
	if !strings.Contains(err.Error(), "snapshotjobs.nvidia.com") {
		t.Errorf("error %q should name the missing API", err)
	}
	requireNoLeakedNamespace(t, f.clientset)
}

func TestCheckSnapshotRestore_FailsWhenSnapshotJobFails(t *testing.T) {
	f := newSnapshotFake(t, func(f *snapshotFake) {
		f.jobStatus = func(job *unstructured.Unstructured) {
			_ = unstructured.SetNestedSlice(job.Object, snapshotConditionList(
				snapshotCond("Running", "False", "JobFailed", ""),
				snapshotCond("Captured", "False", "CaptureFailed", ""),
				snapshotCond("Completed", "False", "JobFailed", ""),
				snapshotCond("Failed", "True", "CheckpointFailed", "while dumping CUDA state: cuda-checkpoint exited 1"),
			), "status", "conditions")
		}
		f.sourcePod = func(namespace, jobName string) *corev1.Pod {
			return &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: jobName + "-abcde", Namespace: namespace, Labels: map[string]string{"job-name": jobName}},
				Status: corev1.PodStatus{
					Phase: corev1.PodFailed,
					ContainerStatuses: []corev1.ContainerStatus{{
						Name: snapshotTargetContainer, State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
							ExitCode: 128, Reason: "StartError", Message: `exec: "python3": executable file not found in $PATH`,
						}},
					}},
				},
			}
		}
	})
	// A generous timeout: the Failed condition must end the wait, not the clock.
	snapshotCaptureTimeout = time.Minute

	start := time.Now()
	out, err := f.run(t)
	if err == nil {
		t.Fatalf("CheckSnapshotRestore() = nil, want an error\n%s", out)
	}
	if time.Since(start) > 30*time.Second {
		t.Error("a Failed SnapshotJob must end the capture wait at once")
	}
	errorstest.WantReportedCode(t, err, errors.ErrCodeInternal)
	if !strings.Contains(err.Error(), "failed (CheckpointFailed)") {
		t.Errorf("error %q should say the SnapshotJob failed, not that it timed out", err)
	}
	for _, want := range []string{"CheckpointFailed", "while dumping CUDA state: cuda-checkpoint exited 1", "main exited 128 (StartError)", "executable file not found"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q lacks %q", err, want)
		}
	}
	if len(f.pods) != 0 {
		t.Error("a failed capture must not go on to create a restore pod")
	}
	requireNoLeakedNamespace(t, f.clientset)
}

func TestCheckSnapshotRestore_CaptureTimeoutNamesTheSourcePod(t *testing.T) {
	f := newSnapshotFake(t, func(f *snapshotFake) {
		f.jobStatus = func(job *unstructured.Unstructured) {
			_ = unstructured.SetNestedSlice(job.Object, snapshotConditionList(
				snapshotCond("Running", "False", "Pending", ""),
			), "status", "conditions")
		}
		// A pod that is pending without a reason the check treats as fatal.
		f.sourcePod = func(namespace, jobName string) *corev1.Pod {
			return &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: jobName + "-abcde", Namespace: namespace, Labels: map[string]string{"job-name": jobName}},
				Status: corev1.PodStatus{
					Phase: corev1.PodPending,
					ContainerStatuses: []corev1.ContainerStatus{{
						Name: snapshotTargetContainer, State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ContainerCreating"}},
					}},
				},
			}
		}
	})
	snapshotCaptureTimeout = 50 * time.Millisecond

	out, err := f.run(t)
	if err == nil {
		t.Fatalf("CheckSnapshotRestore() = nil, want a timeout\n%s", out)
	}
	errorstest.WantReportedCode(t, err, errors.ErrCodeTimeout)
	for _, want := range []string{"did not complete", snapshotProbeImage, "source pod", "ContainerCreating"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q lacks %q", err, want)
		}
	}
	requireNoLeakedNamespace(t, f.clientset)
}

func TestCheckSnapshotRestore_SourcePodFailureEndsTheWaitWithItsLogs(t *testing.T) {
	f := newSnapshotFake(t, func(f *snapshotFake) {
		f.jobStatus = func(job *unstructured.Unstructured) {
			_ = unstructured.SetNestedSlice(job.Object, snapshotConditionList(snapshotCond("Running", "False", "Pending", "")), "status", "conditions")
		}
		f.sourcePod = func(namespace, jobName string) *corev1.Pod {
			return &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: jobName + "-abcde", Namespace: namespace, Labels: map[string]string{"job-name": jobName}},
				Status: corev1.PodStatus{
					Phase: corev1.PodFailed,
					ContainerStatuses: []corev1.ContainerStatus{{
						Name: snapshotTargetContainer, State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1, Reason: "Error", Message: `exec: "python3": executable file not found in $PATH`}},
					}},
				},
			}
		}
	})
	serveProbeLogs(f.clientset, "probe error: libcuda.so.1 not found, so the container runtime injected no NVIDIA driver\n")
	// A generous timeout: the failed pod must end the wait, not the clock.
	snapshotCaptureTimeout = time.Minute

	start := time.Now()
	out, err := f.run(t)
	if err == nil {
		t.Fatalf("CheckSnapshotRestore() = nil, want an error\n%s", out)
	}
	if time.Since(start) > 30*time.Second {
		t.Error("a failed source pod must end the capture wait at once")
	}
	for _, want := range []string{"failed before it was checkpointed", "exited 1", "executable file not found", "libcuda.so.1 not found"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q lacks %q", err, want)
		}
	}
}

func TestCheckSnapshotRestore_RestoreFailures(t *testing.T) {
	tests := []struct {
		name    string
		status  func(pod *corev1.Pod)
		reader  func(*validators.Context, *snapshotRestoreRun) (string, error)
		wantErr []string
	}{
		{
			name: "restore refused by the node agent",
			status: func(pod *corev1.Pod) {
				pod.Status.Phase = corev1.PodRunning
				pod.Status.Conditions = []corev1.PodCondition{
					{Type: corev1.PodReady, Status: corev1.ConditionFalse},
					{Type: corev1.PodConditionType(snapshotRestoredCondition), Status: corev1.ConditionFalse, Reason: "RestoreIncompatible", Message: "while checking compatibility: kernel mismatch"},
				}
			},
			wantErr: []string{"RestoreIncompatible", "while checking compatibility: kernel mismatch"},
		},
		{
			name: "restore failed",
			status: func(pod *corev1.Pod) {
				pod.Status.Phase = corev1.PodRunning
				pod.Status.Conditions = []corev1.PodCondition{
					{Type: corev1.PodConditionType(snapshotRestoredCondition), Status: corev1.ConditionFalse, Reason: "RestoreFailed", Message: "while restoring CUDA state: boom"},
				}
			},
			wantErr: []string{"RestoreFailed", "while restoring CUDA state: boom"},
		},
		{
			name: "restore pod ended",
			status: func(pod *corev1.Pod) {
				pod.Status.Phase = corev1.PodFailed
				pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
					Name: snapshotTargetContainer, State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 112, Reason: "Error"}},
				}}
			},
			wantErr: []string{"the pod ended", "main exited 112", "restore proxy exits 112"},
		},
		{
			name: "restored process reports a mismatch",
			status: func(pod *corev1.Pod) {
				pod.Status.Phase = corev1.PodRunning
				pod.Status.Conditions = []corev1.PodCondition{
					{Type: corev1.PodReady, Status: corev1.ConditionFalse},
					{Type: corev1.PodConditionType(snapshotRestoredCondition), Status: corev1.ConditionTrue, Reason: "RestoreSucceeded"},
				}
			},
			wantErr: []string{"was not Ready", "did not report the expected result", "Restored=True"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newSnapshotFake(t, func(f *snapshotFake) { f.restoreStatus = tt.status })
			snapshotRestoreTimeout = 50 * time.Millisecond
			snapshotProbeResultReader = func(*validators.Context, *snapshotRestoreRun) (string, error) {
				return "mismatch deadbeef", nil
			}
			out, err := f.run(t)
			if err == nil {
				t.Fatalf("CheckSnapshotRestore() = nil, want an error\n%s", out)
			}
			for _, want := range tt.wantErr {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q lacks %q", err, want)
				}
			}
			// The restore pod's own result is in the evidence, however it ended.
			if !strings.Contains(out, "--- Snapshot restore pod ---") || !strings.Contains(out, `probe-result: "mismatch deadbeef"`) {
				t.Errorf("stdout must record the restore pod and what the restored process wrote:\n%s", out)
			}
			requireNoLeakedNamespace(t, f.clientset)
		})
	}
}

func TestCheckSnapshotRestore_PassesWhenTheResultCannotBeRead(t *testing.T) {
	// The readiness gate is the verdict; a cluster that refuses pods/exec must not
	// turn a restore that passed it into a failure.
	f := newSnapshotFake(t)
	snapshotProbeResultReader = func(*validators.Context, *snapshotRestoreRun) (string, error) {
		return "", stderrors.New("pods/exec is forbidden")
	}
	out, err := f.run(t)
	if err != nil {
		t.Fatalf("CheckSnapshotRestore() error = %v\n%s", err, out)
	}
	for _, want := range []string{"not read (pods/exec is forbidden)", "verdict: PASS"} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout missing %q:\n%s", want, out)
		}
	}
}

func TestCheckSnapshotRestore_WithoutNvidiaRuntimeClass(t *testing.T) {
	f := newSnapshotFake(t)
	if err := f.clientset.NodeV1().RuntimeClasses().Delete(context.Background(), nvidiaRuntimeClass, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete RuntimeClass: %v", err)
	}
	if out, err := f.run(t); err != nil {
		t.Fatalf("CheckSnapshotRestore() error = %v\n%s", err, out)
	}
	if spec := podSpecOf(t, f.jobs[0]); spec.RuntimeClassName != nil {
		t.Errorf("source runtimeClassName = %q, want unset when the cluster has no %s class", *spec.RuntimeClassName, nvidiaRuntimeClass)
	}
	if f.pods[0].Spec.RuntimeClassName != nil {
		t.Errorf("restore runtimeClassName = %q, want unset", *f.pods[0].Spec.RuntimeClassName)
	}
}

func TestCheckSnapshotRestore_HonorsTolerationsAndImageOverride(t *testing.T) {
	f := newSnapshotFake(t)
	t.Setenv(snapshotProbeImageEnv, " mirror.example.com/os/python:3.12-slim ")
	ctx := f.context()
	ctx.Tolerations = []corev1.Toleration{{Key: "dedicated", Value: "gpu", Effect: corev1.TaintEffectNoSchedule}}
	var err error
	out := captureStdout(t, func() { err = CheckSnapshotRestore(ctx) })
	if err != nil {
		t.Fatalf("CheckSnapshotRestore() error = %v\n%s", err, out)
	}
	const want = "mirror.example.com/os/python:3.12-slim"
	if got := probeContainerOf(t, f.jobs[0]).Image; got != want {
		t.Errorf("source image = %q, want %q", got, want)
	}
	if got := f.pods[0].Spec.Containers[0].Image; got != want {
		t.Errorf("restore image = %q, want %q: both pods must run the same image", got, want)
	}
	for name, tolerations := range map[string][]corev1.Toleration{
		"source":  podSpecOf(t, f.jobs[0]).Tolerations,
		"restore": f.pods[0].Spec.Tolerations,
	} {
		if len(tolerations) != 1 || tolerations[0].Key != "dedicated" {
			t.Errorf("%s tolerations = %v, want the context tolerations", name, tolerations)
		}
	}
}

func TestCheckSnapshotRestore_NoClusterIsInvalid(t *testing.T) {
	err := CheckSnapshotRestore(&validators.Context{Ctx: context.Background()})
	errorstest.WantReportedCode(t, err, errors.ErrCodeInvalidRequest)
}

func TestRequireDevicePluginProbe(t *testing.T) {
	usable := &allocmode.Mode{DevicePluginUsable: true, DevicePluginNodes: []string{"node1"}}
	tests := []struct {
		name    string
		policy  string
		mode    *allocmode.Mode
		wantErr string
	}{
		{name: "device plugin node", policy: v1.GPUAllocationPolicyUnspecified, mode: usable},
		{name: "explicit device plugin policy", policy: v1.GPUAllocationPolicyDevicePluginExtendedResource, mode: usable},
		{
			name: "DRA policy", policy: v1.GPUAllocationPolicyDRAResourceClaim, mode: usable,
			wantErr: "allocated through the device plugin",
		},
		{
			name: "no device plugin node", policy: v1.GPUAllocationPolicyUnspecified,
			mode:    &allocmode.Mode{DevicePluginDetail: "no node advertises nvidia.com/gpu"},
			wantErr: "no node advertises nvidia.com/gpu",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := requireDevicePluginProbe(tt.policy, tt.mode)
			switch {
			case tt.wantErr == "" && err != nil:
				t.Fatalf("requireDevicePluginProbe() error = %v", err)
			case tt.wantErr != "":
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("requireDevicePluginProbe() = %v, want an error containing %q", err, tt.wantErr)
				}
				errorstest.WantReportedCode(t, err, errors.ErrCodeUnavailable)
			}
		})
	}
}

func TestSnapshotRestoreFailure(t *testing.T) {
	restored := func(status corev1.ConditionStatus, reason string) corev1.PodCondition {
		return corev1.PodCondition{Type: corev1.PodConditionType(snapshotRestoredCondition), Status: status, Reason: reason, Message: "msg"}
	}
	tests := []struct {
		name string
		pod  corev1.Pod
		want string // substring of the failure; "" means the restore may still succeed
	}{
		{name: "pending", pod: corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodPending}}},
		{
			name: "in progress",
			pod:  corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{restored(corev1.ConditionFalse, "RestoreInProgress")}}},
		},
		{
			name: "succeeded",
			pod:  corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{restored(corev1.ConditionTrue, "RestoreSucceeded")}}},
		},
		{
			name: "failed",
			pod:  corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{restored(corev1.ConditionFalse, "RestoreFailed")}}},
			want: "RestoreFailed: msg",
		},
		{
			name: "incompatible",
			pod:  corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{restored(corev1.ConditionFalse, "RestoreIncompatible")}}},
			want: "RestoreIncompatible: msg",
		},
		{
			name: "partially succeeded",
			pod:  corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{restored(corev1.ConditionFalse, "RestorePartiallySucceeded")}}},
			want: "RestorePartiallySucceeded: msg",
		},
		{
			name: "image will not pull",
			pod: corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{{
				Name: "main", Image: "img", State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ImagePullBackOff", Message: "denied"}},
			}}}},
			want: "ImagePullBackOff",
		},
		{name: "pod ended", pod: corev1.Pod{Status: corev1.PodStatus{Phase: corev1.PodSucceeded}}, want: "the pod ended"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := snapshotRestoreFailure(&tt.pod)
			if tt.want == "" && got != "" {
				t.Errorf("snapshotRestoreFailure() = %q, want no failure", got)
			}
			if tt.want != "" && !strings.Contains(got, tt.want) {
				t.Errorf("snapshotRestoreFailure() = %q, want it to contain %q", got, tt.want)
			}
		})
	}
}

func TestSnapshotConditions(t *testing.T) {
	obj := &unstructured.Unstructured{Object: map[string]any{"status": map[string]any{"conditions": snapshotConditionList(
		snapshotCond("Completed", "True", "JobCompleted", "done"),
		snapshotCond("Failed", "False", "NoFailure", ""),
	)}}}
	got := snapshotConditions(obj)
	if c := got["Completed"]; c.Status != "True" || c.Reason != "JobCompleted" || c.Message != "done" {
		t.Errorf("Completed = %+v", c)
	}
	if c, ok := got["Running"]; ok {
		t.Errorf("Running = %+v, want absent", c)
	}
	if len(snapshotConditions(nil)) != 0 || len(snapshotConditions(&unstructured.Unstructured{Object: map[string]any{}})) != 0 {
		t.Error("a missing object or status must yield no conditions")
	}
}

func TestResolveSnapshotProbeImage(t *testing.T) {
	t.Setenv(snapshotProbeImageEnv, "")
	if got := resolveSnapshotProbeImage(); got != snapshotProbeImage {
		t.Errorf("default image = %q, want %q", got, snapshotProbeImage)
	}
	t.Setenv(snapshotProbeImageEnv, "  registry.example.com/mirror/python:3.12  ")
	if got := resolveSnapshotProbeImage(); got != "registry.example.com/mirror/python:3.12" {
		t.Errorf("override image = %q, want the trimmed reference verbatim", got)
	}
	if note := snapshotProbeImageOverrideNote(); note != snapshotProbeImageEnv {
		t.Errorf("override note = %q, want %q", note, snapshotProbeImageEnv)
	}
}

func TestSnapshotProbeScript(t *testing.T) {
	for _, want := range []string{
		"ready-for-snapshot", "restore-complete", "probe-result", "libcuda.so.1",
		"cuMemAlloc_v2", "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2", "cuCtxSynchronize",
		"AICR_PROBE_TOKEN", "SNAPSHOT_CONTROL_DIR",
	} {
		if !strings.Contains(snapshotProbeScript, want) {
			t.Errorf("probe script lacks %q", want)
		}
	}
	// The script's last output line is what the readiness gate compares.
	if !strings.Contains(snapshotProbeScript, `"%s %s\n" % (verdict, token)`) {
		t.Error("probe script must write `<verdict> <token>` to probe-result")
	}
	// A syntax error here would only surface on a GPU node, so compile it when a
	// Python is around.
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not found; skipping the probe script syntax check")
	}
	cmd := exec.CommandContext(context.Background(), python, "-I", "-c", "import ast, sys; ast.parse(sys.stdin.read())")
	cmd.Stdin = strings.NewReader(snapshotProbeScript)
	if out, runErr := cmd.CombinedOutput(); runErr != nil {
		t.Fatalf("probe script does not parse: %v\n%s", runErr, out)
	}
}

func TestRandomProbeTokensDiffer(t *testing.T) {
	a, err := randomProbeToken()
	if err != nil {
		t.Fatalf("randomProbeToken() error = %v", err)
	}
	b, err := randomProbeToken()
	if err != nil {
		t.Fatalf("randomProbeToken() error = %v", err)
	}
	if len(a) != 16 || len(b) != 16 || a == b {
		t.Errorf("tokens %q and %q must be distinct 16-character hex strings", a, b)
	}
}

func TestSnapshotComponentNamespace(t *testing.T) {
	ctx := &validators.Context{ValidationInput: &v1.ValidationInput{ComponentRefs: []recipe.ComponentRef{{Name: snapshotComponentName, Namespace: "custom"}}}}
	if got := snapshotComponentNamespace(ctx); got != "custom" {
		t.Errorf("namespace = %q, want the componentRef's", got)
	}
	ctx.ValidationInput.ComponentRefs[0].Namespace = ""
	if got := snapshotComponentNamespace(ctx); got != snapshotDefaultNamespace {
		t.Errorf("namespace = %q, want the default %q", got, snapshotDefaultNamespace)
	}
	if got := snapshotComponentNamespace(&validators.Context{}); got != snapshotDefaultNamespace {
		t.Errorf("namespace without a validation input = %q, want the default", got)
	}
}

// TestSnapshotProtocolLiterals pins the strings Snapshot defines (its API
// reference and workload contract). A change to one is a protocol change and
// must not slip through as a rename: the code and the tests would otherwise
// follow each other.
func TestSnapshotProtocolLiterals(t *testing.T) {
	tests := []struct{ name, got, want string }{
		{"restore annotation", snapshotRestoreFromAnnotation, "nvidia.com/restore-from"},
		{"restored condition", snapshotRestoredCondition, "nvidia.com/Restored"},
		{"seccomp profile", snapshotSeccompProfile, "profiles/block-iouring.json"},
		{"control directory", snapshotControlDir, "/snapshot-control"},
		{"control directory variable", snapshotControlEnv, "SNAPSHOT_CONTROL_DIR"},
		{"control volume", snapshotControlVolume, "snapshot-control"},
		{"default target container", snapshotTargetContainer, "main"},
		{"RuntimeClass", nvidiaRuntimeClass, "nvidia"},
		{"restore failed reason", snapshotRestoreFailedReason, "RestoreFailed"},
		{"restore incompatible reason", snapshotRestoreIncompatReason, "RestoreIncompatible"},
		{"restore partial reason", snapshotRestorePartialReason, "RestorePartiallySucceeded"},
		{"SnapshotJob resource", snapshotJobGVR.String(), "nvidia.com/v1alpha1, Resource=snapshotjobs"},
		{"PodSnapshot resource", podSnapshotGVR.String(), "nvidia.com/v1alpha1, Resource=podsnapshots"},
		{"PodSnapshotContent resource", podSnapshotContentGVR.String(), "nvidia.com/v1alpha1, Resource=podsnapshotcontents"},
		{"probe image variable", snapshotProbeImageEnv, "AICR_SNAPSHOT_PROBE_IMAGE"},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s = %q, want %q", tt.name, tt.got, tt.want)
		}
	}
}

func TestVerifyPodSnapshot(t *testing.T) {
	newSnap := func(ready, contentName string) *unstructured.Unstructured {
		status := map[string]any{"conditions": snapshotConditionList(snapshotCond("Ready", ready, "R", "not durable"))}
		if contentName != "" {
			status["boundSnapshotContentName"] = contentName
		}
		return &unstructured.Unstructured{Object: map[string]any{
			"apiVersion": "nvidia.com/v1alpha1", "kind": "PodSnapshot",
			"metadata": map[string]any{"name": "snap", "namespace": "ns"}, "status": status,
		}}
	}
	content := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1alpha1", "kind": "PodSnapshotContent",
		"metadata": map[string]any{"name": "content"},
		"spec":     map[string]any{"source": map[string]any{"nodeName": "node7"}},
	}}
	build := func(objs ...*unstructured.Unstructured) *validators.Context {
		dyn := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(),
			map[schema.GroupVersionResource]string{podSnapshotGVR: "PodSnapshotList", podSnapshotContentGVR: "PodSnapshotContentList"})
		for _, o := range objs {
			gvr, ns := podSnapshotGVR, "ns"
			if o.GetKind() == "PodSnapshotContent" {
				gvr, ns = podSnapshotContentGVR, ""
			}
			if err := dyn.Tracker().Create(gvr, o, ns); err != nil {
				t.Fatalf("add %s: %v", o.GetKind(), err)
			}
		}
		return &validators.Context{Ctx: context.Background(), DynamicClient: dyn}
	}
	run := func(ctx *validators.Context) (string, error) {
		return verifyPodSnapshot(ctx, ctx.DynamicClient, "ns", "snap")
	}

	t.Run("ready snapshot names the capture node", func(t *testing.T) {
		node, err := run(build(newSnap("True", "content"), content))
		if err != nil || node != "node7" {
			t.Errorf("verifyPodSnapshot() = %q, %v, want node7, nil", node, err)
		}
	})
	t.Run("not ready is an error with the reason", func(t *testing.T) {
		_, err := run(build(newSnap("False", "content"), content))
		if err == nil || !strings.Contains(err.Error(), "not Ready") || !strings.Contains(err.Error(), "not durable") {
			t.Errorf("verifyPodSnapshot() = %v, want a not-Ready error carrying the condition message", err)
		}
	})
	t.Run("missing snapshot is an error", func(t *testing.T) {
		if _, err := run(build()); err == nil {
			t.Error("verifyPodSnapshot() = nil, want an error for a PodSnapshot that does not exist")
		}
	})
	t.Run("unbound or unreadable content only loses the node", func(t *testing.T) {
		for name, ctx := range map[string]*validators.Context{
			"no bound content": build(newSnap("True", "")),
			"content missing":  build(newSnap("True", "content")),
		} {
			node, err := run(ctx)
			if err != nil || node != "" {
				t.Errorf("%s: verifyPodSnapshot() = %q, %v, want empty node and no error", name, node, err)
			}
		}
	})
}

func TestCheckSnapshotRestore_PinsTheRestoreToTheCaptureNode(t *testing.T) {
	f := newSnapshotFake(t)
	// A second device-plugin node. The checkpoint was taken on node2, so the
	// restore may only go there, while the source could run on either.
	if _, err := f.clientset.CoreV1().Nodes().Create(context.Background(), testNode("node2", withGPUAllocatable("1")), metav1.CreateOptions{}); err != nil {
		t.Fatalf("create node2: %v", err)
	}
	f.dyn.PrependReactor("get", "podsnapshotcontents", func(k8stesting.Action) (bool, runtime.Object, error) {
		return true, &unstructured.Unstructured{Object: map[string]any{
			"apiVersion": "nvidia.com/v1alpha1", "kind": "PodSnapshotContent",
			"metadata": map[string]any{"name": testSnapshotContent},
			"spec":     map[string]any{"source": map[string]any{"nodeName": "node2"}},
		}}, nil
	})
	out, err := f.run(t)
	if err != nil {
		t.Fatalf("CheckSnapshotRestore() error = %v\n%s", err, out)
	}
	requireNodeNameAffinity(t, f.pods[0], []string{"node2"})
	sourceTerms := podSpecOf(t, f.jobs[0]).Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
	if len(sourceTerms) != 2 {
		t.Errorf("source affinity terms = %d, want one per device-plugin node (2)", len(sourceTerms))
	}
	if !strings.Contains(out, "Captured on node: node2") {
		t.Errorf("stdout must record the capture node:\n%s", out)
	}
}

func TestSnapshotSourcePodPicksTheNewestPod(t *testing.T) {
	// The batch Job replaced a failed pod; the failed one sorts last by name but
	// is the older one.
	now := metav1.Now()
	older := metav1.NewTime(now.Add(-time.Minute))
	mk := func(name string, created metav1.Time) *corev1.Pod {
		return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: "ns", CreationTimestamp: created, Labels: map[string]string{"job-name": "job"},
		}}
	}
	clientset := k8sfake.NewClientset(mk("job-zzzzz", older), mk("job-aaaaa", now))
	ctx := &validators.Context{Ctx: context.Background(), Clientset: clientset}
	r := &snapshotRestoreRun{run: &gpuTestRun{namespace: "ns"}, jobName: "job"}

	pod := snapshotSourcePod(context.Background(), ctx, r)
	if pod == nil || pod.Name != "job-aaaaa" {
		t.Errorf("snapshotSourcePod() = %v, want the newest pod job-aaaaa", pod)
	}
	r.jobName = "other"
	if pod = snapshotSourcePod(context.Background(), ctx, r); pod != nil {
		t.Errorf("snapshotSourcePod() = %v for a job with no pods, want nil", pod)
	}
}
