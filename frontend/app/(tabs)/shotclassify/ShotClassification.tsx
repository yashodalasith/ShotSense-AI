/**
 * Shot Classification Screen - React Native
 * Complete 3D Human Avatar with Biomechanical Indicators
 */

import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Modal,
  Dimensions,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import { GLView } from "expo-gl";
import * as THREE from "three";
import { Renderer } from "expo-three";
import Animated, {
  FadeInUp,
  FadeInDown,
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
} from "react-native-reanimated";

import {
  getShotTypes,
  analyzeShot,
  ShotType,
  AnalysisResult,
} from "../../../services/shotClassificationApi";

const { width: SCREEN_WIDTH } = Dimensions.get("window");

// 3D Avatar Component with Human Mannequin
interface Keypoint3D {
  joint: string;
  index: number;
  position: { x: number; y: number; z: number };
}

interface Mistake {
  joint_id: string;
  body_part: string;
  severity: "critical" | "major" | "minor" | "negligible";
  severity_color: string;
  glow_intensity: number;
  explanation: string;
  recommendation: string;
}

interface Avatar3DProps {
  keypoints: Keypoint3D[];
  mistakes: Mistake[];
  isPrototype: boolean;
}

const Avatar3D: React.FC<Avatar3DProps> = ({
  keypoints,
  mistakes,
  isPrototype,
}) => {
  const glViewRef = useRef(null);

  const onContextCreate = async (gl: WebGLRenderingContext) => {
    const { drawingBufferWidth: width, drawingBufferHeight: height } = gl;

    const renderer = new Renderer({ gl });
    renderer.setSize(width, height);
    renderer.setClearColor(0xf8f9ff);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 0, 250);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
    keyLight.position.set(100, 200, 150);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x6366f1, 0.4);
    fillLight.position.set(-100, 50, -50);
    scene.add(fillLight);

    // Calculate center and scale
    const center = new THREE.Vector3(0, 0, 0);
    keypoints.forEach((kp) => {
      center.x += kp.position.x;
      center.y += kp.position.y;
      center.z += kp.position.z;
    });
    center.divideScalar(keypoints.length);

    let maxDistance = 0;
    keypoints.forEach((kp) => {
      const dist = new THREE.Vector3(
        kp.position.x,
        kp.position.y,
        kp.position.z
      ).distanceTo(center);
      if (dist > maxDistance) maxDistance = dist;
    });
    const SCALE = 80 / maxDistance;

    // Helper to get scaled position
    const getScaledPos = (kp: Keypoint3D) =>
      new THREE.Vector3(
        (kp.position.x - center.x) * SCALE,
        (kp.position.y - center.y) * SCALE,
        (kp.position.z - center.z) * SCALE
      );

    // Get joint by name
    const getJoint = (name: string) => keypoints.find((k) => k.joint === name);

    // Create human body parts
    const bodyColor = isPrototype ? 0x10b981 : 0x6366f1;
    const bodyMaterial = new THREE.MeshPhongMaterial({
      color: bodyColor,
      transparent: true,
      opacity: isPrototype ? 0.5 : 0.85,
      emissive: isPrototype ? 0x059669 : 0x4f46e5,
      emissiveIntensity: 0.2,
    });

    // HEAD
    const nose = getJoint("nose");
    if (nose) {
      const headPos = getScaledPos(nose);
      const headGeometry = new THREE.SphereGeometry(8, 16, 16);
      const head = new THREE.Mesh(headGeometry, bodyMaterial);
      head.position.copy(headPos);
      scene.add(head);
    }

    // TORSO
    const leftShoulder = getJoint("left_shoulder");
    const rightShoulder = getJoint("right_shoulder");
    const leftHip = getJoint("left_hip");
    const rightHip = getJoint("right_hip");

    if (leftShoulder && rightShoulder && leftHip && rightHip) {
      const lsPos = getScaledPos(leftShoulder);
      const rsPos = getScaledPos(rightShoulder);
      const lhPos = getScaledPos(leftHip);
      const rhPos = getScaledPos(rightHip);

      const torsoCenter = new THREE.Vector3()
        .addVectors(lsPos, rsPos)
        .add(lhPos)
        .add(rhPos)
        .multiplyScalar(0.25);

      const torsoHeight = lsPos.distanceTo(lhPos);
      const torsoWidth = lsPos.distanceTo(rsPos);

      const torsoGeometry = new THREE.BoxGeometry(
        torsoWidth * 1.2,
        torsoHeight,
        torsoWidth * 0.6
      );
      const torso = new THREE.Mesh(torsoGeometry, bodyMaterial);
      torso.position.copy(torsoCenter);
      scene.add(torso);
    }

    // LIMBS - Create cylinders between joints
    const createLimb = (
      start: Keypoint3D | undefined,
      end: Keypoint3D | undefined,
      radius: number
    ) => {
      if (!start || !end) return;

      const startPos = getScaledPos(start);
      const endPos = getScaledPos(end);

      const direction = new THREE.Vector3().subVectors(endPos, startPos);
      const length = direction.length();
      const midpoint = new THREE.Vector3()
        .addVectors(startPos, endPos)
        .multiplyScalar(0.5);

      const geometry = new THREE.CylinderGeometry(radius, radius, length, 8, 1);
      const limb = new THREE.Mesh(geometry, bodyMaterial);

      limb.position.copy(midpoint);
      limb.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.normalize()
      );

      scene.add(limb);
    };

    // Arms
    createLimb(leftShoulder, getJoint("left_elbow"), 2.5);
    createLimb(getJoint("left_elbow"), getJoint("left_wrist"), 2);
    createLimb(rightShoulder, getJoint("right_elbow"), 2.5);
    createLimb(getJoint("right_elbow"), getJoint("right_wrist"), 2);

    // Legs
    createLimb(leftHip, getJoint("left_knee"), 3);
    createLimb(getJoint("left_knee"), getJoint("left_ankle"), 2.5);
    createLimb(rightHip, getJoint("right_knee"), 3);
    createLimb(getJoint("right_knee"), getJoint("right_ankle"), 2.5);

    // MISTAKE INDICATORS - Only if not prototype
    if (!isPrototype && mistakes.length > 0) {
      const mistakeMap = new Map(
        mistakes.map((m) => [m.joint_id.toLowerCase(), m])
      );

      // Joint mapping for mistake IDs to actual joint names
      const jointMapping: { [key: string]: string } = {
        back_elbow: "left_elbow",
        front_elbow: "right_elbow",
        back_wrist: "left_wrist",
        front_wrist: "right_wrist",
        torso_bend: "nose",
        left_hip: "left_hip",
        right_hip: "right_hip",
      };

      mistakes.forEach((mistake) => {
        const jointName = jointMapping[mistake.joint_id] || mistake.joint_id;
        const joint = getJoint(jointName);

        if (joint) {
          const pos = getScaledPos(joint);

          // Glowing sphere at mistake location
          const glowGeometry = new THREE.SphereGeometry(6, 16, 16);
          const glowMaterial = new THREE.MeshBasicMaterial({
            color: new THREE.Color(mistake.severity_color),
            transparent: true,
            opacity: 0.7,
          });
          const glow = new THREE.Mesh(glowGeometry, glowMaterial);
          glow.position.copy(pos);
          scene.add(glow);

          // Outer glow ring
          const ringGeometry = new THREE.SphereGeometry(10, 16, 16);
          const ringMaterial = new THREE.MeshBasicMaterial({
            color: new THREE.Color(mistake.severity_color),
            transparent: true,
            opacity: 0.3,
          });
          const ring = new THREE.Mesh(ringGeometry, ringMaterial);
          ring.position.copy(pos);
          scene.add(ring);

          // Arrow pointing to mistake
          const arrowDir = new THREE.Vector3(1, 1, 0.5).normalize();
          const arrowLength = 20;
          const arrowStart = pos
            .clone()
            .add(arrowDir.clone().multiplyScalar(15));

          const arrowGeometry = new THREE.ConeGeometry(2, 8, 8);
          const arrowMaterial = new THREE.MeshBasicMaterial({
            color: new THREE.Color(mistake.severity_color),
          });
          const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
          arrow.position.copy(arrowStart);
          arrow.lookAt(pos);
          arrow.rotateX(Math.PI / 2);
          scene.add(arrow);

          // Line from arrow to joint
          const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            arrowStart,
            pos,
          ]);
          const lineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color(mistake.severity_color),
            linewidth: 2,
          });
          const line = new THREE.Line(lineGeometry, lineMaterial);
          scene.add(line);
        }
      });
    }

    // Ground plane (cricket pitch)
    const groundGeometry = new THREE.PlaneGeometry(200, 200);
    const groundMaterial = new THREE.MeshPhongMaterial({
      color: 0x86efac,
      transparent: true,
      opacity: 0.3,
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -80;
    scene.add(ground);

    // Crease lines
    const creaseGeometry = new THREE.PlaneGeometry(2, 100);
    const creaseMaterial = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.6,
    });
    const crease = new THREE.Mesh(creaseGeometry, creaseMaterial);
    crease.rotation.x = -Math.PI / 2;
    crease.position.y = -79.5;
    scene.add(crease);

    // Cricket bat (if not prototype)
    if (!isPrototype) {
      const rightWrist = getJoint("right_wrist");
      if (rightWrist) {
        const wristPos = getScaledPos(rightWrist);

        // Bat blade
        const batBladeGeometry = new THREE.BoxGeometry(2, 25, 8);
        const batBladeMaterial = new THREE.MeshPhongMaterial({
          color: 0xd4a574,
          emissive: 0x8b6f47,
          emissiveIntensity: 0.2,
        });
        const batBlade = new THREE.Mesh(batBladeGeometry, batBladeMaterial);
        batBlade.position.copy(wristPos);
        batBlade.position.y -= 15;
        scene.add(batBlade);

        // Bat handle
        const batHandleGeometry = new THREE.CylinderGeometry(1, 1, 15, 8);
        const batHandleMaterial = new THREE.MeshPhongMaterial({
          color: 0x1e293b,
        });
        const batHandle = new THREE.Mesh(batHandleGeometry, batHandleMaterial);
        batHandle.position.copy(wristPos);
        batHandle.position.y += 8;
        scene.add(batHandle);
      }
    }

    // Animation - subtle rotation
    let angle = isPrototype ? Math.PI / 4 : 0;
    let time = 0;
    const render = () => {
      requestAnimationFrame(render);

      time += 0.01;
      angle += 0.003;

      // Gentle rotation
      camera.position.x = Math.sin(angle) * 250;
      camera.position.z = Math.cos(angle) * 250;

      // Subtle up-down sway
      camera.position.y = Math.sin(time * 0.5) * 10;

      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      (gl as any).endFrameEXP();
    };
    render();
  };

  return (
    <GLView
      ref={glViewRef}
      style={styles.avatar3D}
      onContextCreate={onContextCreate}
    />
  );
};

export default function ShotClassificationScreen() {
  const [shotTypes, setShotTypes] = useState<ShotType[]>([]);
  const [selectedShot, setSelectedShot] = useState<string | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showMistakeModal, setShowMistakeModal] = useState<Mistake | null>(
    null
  );
  const [score, setScore] = useState(0);

  const scoreAnim = useSharedValue(0);
  const buttonScale = useSharedValue(1);

  useEffect(() => {
    loadShotTypes();
    ImagePicker.requestMediaLibraryPermissionsAsync();
  }, []);

  useEffect(() => {
    if (result) {
      scoreAnim.value = withTiming(result.intent_score, { duration: 1500 });

      let start = 0;
      const end = result.intent_score;
      const duration = 1500;
      const startTime = Date.now();

      const animate = () => {
        const now = Date.now();
        const progress = Math.min((now - startTime) / duration, 1);
        setScore(Math.floor(start + (end - start) * progress));

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };
      animate();
    }
  }, [result]);

  const loadShotTypes = async () => {
    try {
      setShotTypes(await getShotTypes());
    } catch {
      Alert.alert("Error", "Failed to load shot types");
    }
  };

  const pickVideo = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: "videos",
      quality: 1,
    });

    if (!res.canceled && res.assets.length > 0) {
      setVideoUri(res.assets[0].uri);
      setResult(null);
      setScore(0);
    }
  };

  const handleAnalyze = async () => {
    if (!videoUri || !selectedShot) return;

    buttonScale.value = withSpring(0.95, {}, () => {
      buttonScale.value = withSpring(1);
    });

    try {
      setAnalyzing(true);
      const analysisResult = await analyzeShot(videoUri, selectedShot);
      setResult(analysisResult);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Analysis failed";
      Alert.alert("Analysis Failed", message);
    } finally {
      setAnalyzing(false);
    }
  };

  const buttonAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
  }));

  function capitalize(text: string) {
    if (!text) return "";
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Hero Section */}
      <LinearGradient
        colors={["rgba(99, 102, 241, 0.15)", "rgba(139, 92, 246, 0.1)"]}
        style={styles.hero}
      >
        <Animated.View entering={FadeInDown.duration(800)}>
          <Text style={styles.heroTitle}>SHOT INTELLIGENCE</Text>
          <Text style={styles.heroSubtitle}>
            AI-Powered Cricket Biomechanics
          </Text>
        </Animated.View>

        <View style={styles.heroStats}>
          {[
            { value: "99.2%", label: "Accuracy" },
            { value: "3D", label: "Visualization" },
            { value: "AI", label: "Powered" },
          ].map((stat, idx) => (
            <Animated.View
              key={idx}
              entering={FadeInUp.delay(idx * 100).duration(600)}
              style={styles.stat}
            >
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </Animated.View>
          ))}
        </View>
      </LinearGradient>

      {/* Shot Selection */}
      <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
        <Text style={styles.cardTitle}>Select Your Intended Shot</Text>
        <Text style={styles.cardSubtitle}>
          Choose the shot you're attempting
        </Text>

        <View style={styles.shotGrid}>
          {shotTypes.map((shot) => {
            const isActive = selectedShot === shot.value;
            return (
              <TouchableOpacity
                key={shot.value}
                onPress={() => setSelectedShot(shot.value)}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={
                    isActive
                      ? ["#6366f1", "#8b5cf6"]
                      : ["rgba(99, 102, 241, 0.08)", "rgba(139, 92, 246, 0.06)"]
                  }
                  style={[styles.shotChip, isActive && styles.shotChipActive]}
                >
                  <Text
                    style={[
                      styles.shotChipText,
                      isActive && styles.shotChipTextActive,
                    ]}
                  >
                    {shot.label}
                  </Text>
                  {isActive && <Text style={styles.checkMark}>✓</Text>}
                </LinearGradient>
              </TouchableOpacity>
            );
          })}
        </View>
      </Animated.View>

      {/* Video Upload */}
      <Animated.View entering={FadeInUp.delay(300)} style={styles.card}>
        <TouchableOpacity onPress={pickVideo} activeOpacity={0.8}>
          <LinearGradient
            colors={
              videoUri
                ? ["#10b981", "#34d399"]
                : ["rgba(99, 102, 241, 0.08)", "rgba(139, 92, 246, 0.08)"]
            }
            style={styles.uploadArea}
          >
            <Text style={styles.uploadIcon}>{videoUri ? "✓" : "📹"}</Text>
            <View style={styles.uploadText}>
              <Text style={styles.uploadTitle}>
                {videoUri ? "Video Ready" : "Upload Your Shot"}
              </Text>
              <Text style={styles.uploadSubtitle}>
                Slow-motion video preferred
              </Text>
            </View>
          </LinearGradient>
        </TouchableOpacity>
      </Animated.View>

      {/* Analyze Button */}
      <Animated.View style={[styles.analyzeButtonContainer, buttonAnimStyle]}>
        <TouchableOpacity
          disabled={!videoUri || !selectedShot || analyzing}
          onPress={handleAnalyze}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={["#6366f1", "#8b5cf6"]}
            style={[
              styles.analyzeButton,
              (!videoUri || !selectedShot || analyzing) &&
                styles.analyzeButtonDisabled,
            ]}
          >
            {analyzing ? (
              <>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.analyzeButtonText}>
                  Analyzing Biomechanics...
                </Text>
              </>
            ) : (
              <>
                <Text style={styles.analyzeIcon}>🚀</Text>
                <Text style={styles.analyzeButtonText}>Run AI Analysis</Text>
              </>
            )}
          </LinearGradient>
        </TouchableOpacity>
      </Animated.View>

      {/* Results */}
      {result && (
        <>
          {/* Score Card */}
          <Animated.View entering={FadeInUp.delay(100)} style={styles.card}>
            <View style={styles.scoreRing}>
              <View style={styles.scoreContent}>
                <Text style={styles.scoreValue}>{score}%</Text>
                <Text style={styles.scoreLabel}>Intent Accuracy</Text>
              </View>
            </View>

            <View style={styles.scoreDetails}>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Intended:</Text>
                <Text style={styles.detailValueIntended}>
                  {result.intended_shot}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Detected:</Text>
                <Text
                  style={[
                    styles.detailValue,
                    result.is_correct
                      ? styles.detailValueCorrect
                      : styles.detailValueIncorrect,
                  ]}
                >
                  {result.predicted_shot}
                </Text>
              </View>
            </View>
          </Animated.View>

          {/* 3D Visualization */}
          <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
            <Text style={styles.cardTitle}>3D Biomechanical Analysis</Text>
            <Text style={styles.cardSubtitle}>
              Your form vs. Perfect prototype
            </Text>

            <View style={styles.avatarContainer}>
              <View style={styles.avatarWrapper}>
                <LinearGradient
                  colors={["#6366f1", "#8b5cf6"]}
                  style={styles.avatarLabel}
                >
                  <Text style={styles.avatarLabelText}>Your Form</Text>
                </LinearGradient>
                <Avatar3D
                  keypoints={result.visual_feedback.keypoints_3d.actual}
                  mistakes={result.visual_feedback.mistakes}
                  isPrototype={false}
                />
              </View>

              <View style={styles.avatarWrapper}>
                <LinearGradient
                  colors={["#10b981", "#34d399"]}
                  style={styles.avatarLabel}
                >
                  <Text style={styles.avatarLabelText}>Perfect Form</Text>
                </LinearGradient>
                <Avatar3D
                  keypoints={result.visual_feedback.keypoints_3d.prototype}
                  mistakes={[]}
                  isPrototype={true}
                />
              </View>
            </View>
          </Animated.View>

          {/* Mistakes */}
          {result.visual_feedback.mistakes.length > 0 && (
            <Animated.View entering={FadeInUp.delay(300)} style={styles.card}>
              <Text style={styles.cardTitle}>Technical Issues Detected</Text>
              <Text style={styles.cardSubtitle}>
                {result.visual_feedback.mistakes.length} biomechanical errors
              </Text>

              <View style={styles.mistakesList}>
                {result.visual_feedback.mistakes.map((mistake, idx) => (
                  <TouchableOpacity
                    key={idx}
                    onPress={() => setShowMistakeModal(mistake)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.mistakeItem}>
                      <View
                        style={[
                          styles.mistakeIndicator,
                          { backgroundColor: mistake.severity_color },
                        ]}
                      />
                      <View style={styles.mistakeContent}>
                        <View style={styles.mistakeHeader}>
                          <Text style={styles.mistakePart}>
                            {mistake.body_part}
                          </Text>
                          <View
                            style={[
                              styles.mistakeSeverity,
                              styles[
                                `mistakeSeverity${capitalize(
                                  mistake.severity
                                )}` as keyof typeof styles
                              ],
                            ]}
                          >
                            <Text
                              style={[
                                styles.mistakeSeverityText,
                                mistake.severity === "minor" &&
                                  styles.mistakeSeverityTextMinor,
                              ]}
                            >
                              {mistake.severity}
                            </Text>
                          </View>
                        </View>
                        <Text style={styles.mistakeExplanation}>
                          {mistake.explanation}
                        </Text>
                      </View>
                      <Text style={styles.mistakeArrow}>→</Text>
                    </View>
                  </TouchableOpacity>
                ))}
              </View>
            </Animated.View>
          )}

          {/* AI Feedback */}
          <Animated.View entering={FadeInUp.delay(400)}>
            <LinearGradient
              colors={["rgba(16, 185, 129, 0.1)", "rgba(52, 211, 153, 0.08)"]}
              style={[styles.card, styles.feedbackCard]}
            >
              <Text style={styles.feedbackIcon}>💬</Text>
              <Text style={styles.feedbackTitle}>AI Coach Feedback</Text>
              <Text style={styles.feedbackText}>
                {result.coaching_feedback}
              </Text>
            </LinearGradient>
          </Animated.View>

          {/* Probability Distribution */}
          <Animated.View entering={FadeInUp.delay(500)} style={styles.card}>
            <Text style={styles.cardTitle}>Shot Classification Confidence</Text>

            <View style={styles.probabilityBars}>
              {Object.entries(result.ensemble_probabilities)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 5)
                .map(([shot, prob], idx) => (
                  <Animated.View
                    key={shot}
                    entering={FadeInUp.delay(600 + idx * 50)}
                    style={styles.probRow}
                  >
                    <Text style={styles.probLabel}>{shot}</Text>
                    <View style={styles.probBarBg}>
                      <LinearGradient
                        colors={["#6366f1", "#8b5cf6"]}
                        style={[
                          styles.probBarFill,
                          { width: `${prob * 100}%` },
                        ]}
                      />
                    </View>
                    <Text style={styles.probValue}>
                      {(prob * 100).toFixed(1)}%
                    </Text>
                  </Animated.View>
                ))}
            </View>
          </Animated.View>
        </>
      )}

      {/* Mistake Modal */}
      <Modal
        visible={!!showMistakeModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowMistakeModal(null)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modal}>
            <TouchableOpacity
              style={styles.modalClose}
              onPress={() => setShowMistakeModal(null)}
            >
              <Text style={styles.modalCloseText}>×</Text>
            </TouchableOpacity>

            {showMistakeModal && (
              <>
                <View style={styles.modalHeader}>
                  <View
                    style={[
                      styles.modalIndicator,
                      {
                        backgroundColor: showMistakeModal.severity_color,
                      },
                    ]}
                  />
                  <Text style={styles.modalTitle}>
                    {showMistakeModal.body_part}
                  </Text>
                  <View
                    style={[
                      styles.mistakeSeverity,
                      styles[
                        `mistakeSeverity${capitalize(
                          showMistakeModal.severity
                        )}` as keyof typeof styles
                      ],
                    ]}
                  >
                    <Text
                      style={[
                        styles.mistakeSeverityText,
                        showMistakeModal.severity === "minor" &&
                          styles.mistakeSeverityTextMinor,
                      ]}
                    >
                      {showMistakeModal.severity}
                    </Text>
                  </View>
                </View>

                <View style={styles.modalBody}>
                  <View style={styles.modalSection}>
                    <Text style={styles.modalSectionTitle}>Problem</Text>
                    <Text style={styles.modalSectionText}>
                      {showMistakeModal.explanation}
                    </Text>
                  </View>
                  <View style={styles.modalSection}>
                    <Text style={styles.modalSectionTitle}>Correction</Text>
                    <Text style={styles.modalSectionText}>
                      {showMistakeModal.recommendation}
                    </Text>
                  </View>
                </View>
              </>
            )}
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#ffffff",
  },

  // Hero Section
  hero: {
    paddingTop: 60,
    paddingBottom: 40,
    paddingHorizontal: 20,
    alignItems: "center",
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: "900",
    color: "#6366f1",
    letterSpacing: 2,
    textAlign: "center",
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 14,
    color: "#8b5cf6",
    textAlign: "center",
    marginBottom: 24,
  },
  heroStats: {
    flexDirection: "row",
    justifyContent: "space-around",
    width: "100%",
    marginTop: 20,
  },
  stat: {
    alignItems: "center",
  },
  statValue: {
    fontSize: 24,
    fontWeight: "900",
    color: "#6366f1",
  },
  statLabel: {
    fontSize: 10,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginTop: 4,
  },

  // Card
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 20,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "rgba(99, 102, 241, 0.15)",
    shadowColor: "#6366f1",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 4,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#1e293b",
    marginBottom: 6,
  },
  cardSubtitle: {
    fontSize: 13,
    color: "#8b5cf6",
    marginBottom: 16,
  },

  // Shot Selection
  shotGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  shotChip: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: "rgba(99, 102, 241, 0.2)",
    minWidth: (SCREEN_WIDTH - 72) / 3,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  shotChipActive: {
    borderColor: "#6366f1",
    shadowColor: "#6366f1",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  shotChipText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#8b5cf6",
  },
  shotChipTextActive: {
    color: "#fff",
  },
  checkMark: {
    position: "absolute",
    top: 4,
    right: 4,
    color: "#fff",
    fontSize: 16,
  },

  // Upload
  uploadArea: {
    flexDirection: "row",
    alignItems: "center",
    padding: 20,
    borderRadius: 16,
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "rgba(99, 102, 241, 0.2)",
  },
  uploadIcon: {
    fontSize: 40,
    marginRight: 16,
  },
  uploadText: {
    flex: 1,
  },
  uploadTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#1e293b",
    marginBottom: 4,
  },
  uploadSubtitle: {
    fontSize: 12,
    color: "#64748b",
  },

  // Analyze Button
  analyzeButtonContainer: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  analyzeButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 18,
    borderRadius: 16,
    shadowColor: "#6366f1",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  analyzeButtonDisabled: {
    opacity: 0.5,
  },
  analyzeIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  analyzeButtonText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#fff",
    letterSpacing: 1,
  },

  // Score Card
  scoreRing: {
    width: 200,
    height: 200,
    alignSelf: "center",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 24,
    borderRadius: 100,
    borderWidth: 12,
    borderColor: "#e0e7ff",
    backgroundColor: "rgba(99, 102, 241, 0.05)",
  },
  scoreContent: {
    alignItems: "center",
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: "900",
    color: "#6366f1",
  },
  scoreLabel: {
    fontSize: 11,
    color: "#8b5cf6",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  scoreDetails: {
    gap: 12,
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
    backgroundColor: "rgba(99, 102, 241, 0.05)",
    borderRadius: 12,
  },
  detailLabel: {
    fontSize: 14,
    color: "#64748b",
    fontWeight: "600",
  },
  detailValue: {
    fontSize: 16,
    fontWeight: "800",
    textTransform: "uppercase",
  },
  detailValueIntended: {
    fontSize: 16,
    fontWeight: "800",
    color: "#6366f1",
    textTransform: "uppercase",
  },
  detailValueCorrect: {
    color: "#10b981",
  },
  detailValueIncorrect: {
    color: "#f59e0b",
  },

  // 3D Avatar
  avatarContainer: {
    gap: 16,
  },
  avatarWrapper: {
    marginBottom: 16,
  },
  avatarLabel: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignSelf: "center",
    marginBottom: 12,
  },
  avatarLabelText: {
    color: "#fff",
    fontWeight: "800",
    fontSize: 14,
  },
  avatar3D: {
    width: "100%",
    height: 350,
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "#f8f9ff",
    borderWidth: 1,
    borderColor: "rgba(99, 102, 241, 0.1)",
  },

  // Mistakes
  mistakesList: {
    gap: 12,
  },
  mistakeItem: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    backgroundColor: "rgba(99, 102, 241, 0.04)",
    borderRadius: 16,
    gap: 12,
    borderWidth: 1,
    borderColor: "rgba(99, 102, 241, 0.08)",
  },
  mistakeIndicator: {
    width: 8,
    height: 60,
    borderRadius: 4,
  },
  mistakeContent: {
    flex: 1,
  },
  mistakeHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  mistakePart: {
    fontSize: 16,
    fontWeight: "800",
    color: "#1e293b",
    flex: 1,
  },
  mistakeSeverity: {
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 12,
    alignSelf: "flex-start",
  },
  mistakeSeverityCritical: {
    backgroundColor: "#ef4444",
  },
  mistakeSeverityMajor: {
    backgroundColor: "#f97316",
  },
  mistakeSeverityMinor: {
    backgroundColor: "#fbbf24",
  },
  mistakeSeverityNegligible: {
    backgroundColor: "#94a3b8",
  },
  mistakeSeverityText: {
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    color: "#fff",
  },
  mistakeSeverityTextMinor: {
    color: "#1e293b",
  },
  mistakeExplanation: {
    fontSize: 14,
    color: "#475569",
    lineHeight: 20,
  },
  mistakeArrow: {
    fontSize: 24,
    color: "#6366f1",
  },

  // Feedback
  feedbackCard: {
    alignItems: "center",
  },
  feedbackIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  feedbackTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#10b981",
    marginBottom: 12,
  },
  feedbackText: {
    fontSize: 15,
    color: "#1e293b",
    lineHeight: 24,
    textAlign: "center",
  },

  // Probability
  probabilityBars: {
    gap: 12,
  },
  probRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  probLabel: {
    width: 100,
    fontSize: 12,
    fontWeight: "700",
    color: "#8b5cf6",
    textTransform: "uppercase",
  },
  probBarBg: {
    flex: 1,
    height: 32,
    backgroundColor: "rgba(99, 102, 241, 0.08)",
    borderRadius: 16,
    overflow: "hidden",
  },
  probBarFill: {
    height: "100%",
    borderRadius: 16,
  },
  probValue: {
    width: 60,
    fontSize: 14,
    fontWeight: "800",
    color: "#1e293b",
    textAlign: "right",
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(15, 23, 42, 0.8)",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  modal: {
    backgroundColor: "#ffffff",
    borderRadius: 24,
    padding: 24,
    paddingRight: 50,
    width: "100%",
    maxWidth: 500,
    borderWidth: 2,
    borderColor: "rgba(99, 102, 241, 0.2)",
    shadowColor: "#6366f1",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    elevation: 12,
  },
  modalClose: {
    position: "absolute",
    top: 12,
    right: 12,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "rgba(239, 68, 68, 0.12)", // 🔴 soft red bg
    justifyContent: "center",
    alignItems: "center",
    zIndex: 20, // ensure above header
  },
  modalCloseText: {
    color: "#ef4444", // 🔴 red cross
    fontSize: 22,
    fontWeight: "900",
    lineHeight: 22, // prevents vertical clipping
  },
  modalHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
    marginBottom: 24,
  },
  modalIndicator: {
    width: 8,
    height: 80,
    borderRadius: 4,
  },
  modalTitle: {
    flex: 1,
    fontSize: 24,
    fontWeight: "900",
    color: "#1e293b",
  },
  modalBody: {
    gap: 20,
  },
  modalSection: {},
  modalSectionTitle: {
    fontSize: 12,
    fontWeight: "800",
    color: "#6366f1",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 8,
  },
  modalSectionText: {
    fontSize: 15,
    color: "#475569",
    lineHeight: 22,
  },
});
