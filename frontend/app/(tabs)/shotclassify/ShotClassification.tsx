/**
 * Shot Classification Screen - React Native
 * Complete 3D Avatar Implementation
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
  Animated as RNAnimated,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import { GLView } from "expo-gl";
import * as THREE from "three";
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

// 3D Avatar Component
interface Keypoint3D {
  joint: string;
  index: number;
  position: { x: number; y: number; z: number };
}

interface Mistake {
  joint_id: string;
  body_part: string;
  severity: "critical" | "major" | "minor";
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

    // Scene setup
    const renderer = new THREE.WebGLRenderer({ context: gl });
    renderer.setSize(width, height);
    renderer.setClearColor(0x0a0a0a);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 0, 800);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xff4081, 1, 1000);
    pointLight.position.set(200, 200, 200);
    scene.add(pointLight);

    // Materials
    const jointMaterial = new THREE.MeshPhongMaterial({
      color: isPrototype ? 0x4caf50 : 0xff4081,
      emissive: isPrototype ? 0x2e7d32 : 0xd81b60,
      transparent: true,
      opacity: isPrototype ? 0.4 : 1,
    });

    const mistakeJoints = new Set(mistakes?.map((m) => m.joint_id) || []);

    // Create joints
    keypoints.forEach((kp) => {
      const isMistake = mistakeJoints.has(kp.joint);
      const geometry = new THREE.SphereGeometry(isMistake ? 12 : 8, 16, 16);

      let material;
      let mistake;
      if (isMistake) {
        mistake = mistakes.find((m) => m.joint_id === kp.joint);
        if (mistake) {
          const color = new THREE.Color(mistake.severity_color);
          material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: mistake.glow_intensity,
            transparent: true,
            opacity: 0.9,
          });
        } else {
          material = jointMaterial;
        }
      } else {
        material = jointMaterial;
      }

      const joint = new THREE.Mesh(geometry, material);
      joint.position.set(kp.position.x - 512, kp.position.y, kp.position.z);
      scene.add(joint);

      // Glow effect for mistakes
      if (isMistake && mistake) {
        const glowGeometry = new THREE.SphereGeometry(18, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
          color: mistake.severity_color,
          transparent: true,
          opacity: 0.2,
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.copy(joint.position);
        scene.add(glow);
      }
    });

    // Draw connections
    const connections = [
      ["left_shoulder", "right_shoulder"],
      ["left_shoulder", "left_elbow"],
      ["left_elbow", "left_wrist"],
      ["right_shoulder", "right_elbow"],
      ["right_elbow", "right_wrist"],
      ["left_shoulder", "left_hip"],
      ["right_shoulder", "right_hip"],
      ["left_hip", "right_hip"],
      ["left_hip", "left_knee"],
      ["left_knee", "left_ankle"],
      ["right_hip", "right_knee"],
      ["right_knee", "right_ankle"],
    ];

    connections.forEach(([from, to]) => {
      const fromKp = keypoints.find((k) => k.joint === from);
      const toKp = keypoints.find((k) => k.joint === to);

      if (fromKp && toKp) {
        const points = [
          new THREE.Vector3(
            fromKp.position.x - 512,
            fromKp.position.y,
            fromKp.position.z
          ),
          new THREE.Vector3(
            toKp.position.x - 512,
            toKp.position.y,
            toKp.position.z
          ),
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
          color: isPrototype ? 0x66bb6a : 0xff4081,
          linewidth: 2,
          transparent: true,
          opacity: isPrototype ? 0.3 : 0.8,
        });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
      }
    });

    // Animation loop
    let angle = 0;
    const render = () => {
      requestAnimationFrame(render);
      angle += 0.005;
      camera.position.x = Math.sin(angle) * 800;
      camera.position.z = Math.cos(angle) * 800;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
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
      // Animate score
      scoreAnim.value = withTiming(result.intent_score, { duration: 1500 });

      // Counter animation
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
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
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

  function capitalize(severity: string) {
    throw new Error("Function not implemented.");
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Hero Section */}
      <LinearGradient
        colors={["rgba(255, 23, 68, 0.2)", "rgba(255, 64, 129, 0.1)"]}
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
                      ? ["#ff1744", "#ff4081"]
                      : ["rgba(255, 23, 68, 0.1)", "rgba(255, 23, 68, 0.05)"]
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
                ? ["#4caf50", "#66bb6a"]
                : ["rgba(255, 23, 68, 0.1)", "rgba(255, 64, 129, 0.1)"]
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
            colors={["#d50000", "#ff1744"]}
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
                  colors={["#ff1744", "#ff4081"]}
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
                  colors={["#4caf50", "#66bb6a"]}
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
                            <Text style={styles.mistakeSeverityText}>
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
              colors={["rgba(76, 175, 80, 0.1)", "rgba(102, 187, 106, 0.1)"]}
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
                        colors={["#ff1744", "#ff4081"]}
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
                    <Text style={styles.mistakeSeverityText}>
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
    backgroundColor: "#0a0a0a",
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
    color: "#ff1744",
    letterSpacing: 2,
    textAlign: "center",
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 14,
    color: "#ff80ab",
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
    color: "#ff4081",
  },
  statLabel: {
    fontSize: 10,
    color: "#999",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginTop: 4,
  },

  // Card
  card: {
    backgroundColor: "rgba(26, 10, 20, 0.6)",
    borderRadius: 20,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "rgba(255, 23, 68, 0.2)",
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#fff",
    marginBottom: 6,
  },
  cardSubtitle: {
    fontSize: 13,
    color: "#ff80ab",
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
    borderColor: "rgba(255, 23, 68, 0.3)",
    minWidth: (SCREEN_WIDTH - 72) / 3,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  shotChipActive: {
    borderColor: "#ff4081",
    shadowColor: "#ff4081",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
  },
  shotChipText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#ff80ab",
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
    borderColor: "rgba(255, 23, 68, 0.3)",
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
    color: "#fff",
    marginBottom: 4,
  },
  uploadSubtitle: {
    fontSize: 12,
    color: "#999",
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
    shadowColor: "#ff1744",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
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
    borderColor: "#2a2a2a",
    backgroundColor: "rgba(255, 23, 68, 0.1)",
  },
  scoreContent: {
    alignItems: "center",
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: "900",
    color: "#ff1744",
  },
  scoreLabel: {
    fontSize: 11,
    color: "#ff80ab",
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
    backgroundColor: "rgba(255, 23, 68, 0.05)",
    borderRadius: 12,
  },
  detailLabel: {
    fontSize: 14,
    color: "#999",
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
    color: "#ff4081",
    textTransform: "uppercase",
  },
  detailValueCorrect: {
    color: "#4caf50",
  },
  detailValueIncorrect: {
    color: "#ff9800",
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
    height: 300,
    borderRadius: 16,
    overflow: "hidden",
    backgroundColor: "#0a0a0a",
  },

  // Mistakes
  mistakesList: {
    gap: 12,
  },
  mistakeItem: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    backgroundColor: "rgba(255, 23, 68, 0.05)",
    borderRadius: 16,
    gap: 12,
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
    color: "#fff",
    flex: 1,
  },
  mistakeSeverity: {
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 12,
    alignSelf: "flex-start",
  },
  mistakeSeverityCritical: {
    backgroundColor: "#e74c3c",
  },
  mistakeSeverityMajor: {
    backgroundColor: "#f39c12",
  },
  mistakeSeverityMinor: {
    backgroundColor: "#f1c40f",
  },
  mistakeSeverityText: {
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    color: "#fff",
  },
  mistakeSeverityTextMinor: {
    color: "#000", // minor has dark text
  },
  mistakeExplanation: {
    fontSize: 14,
    color: "#ccc",
    lineHeight: 20,
  },
  mistakeArrow: {
    fontSize: 24,
    color: "#ff4081",
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
    color: "#4caf50",
    marginBottom: 12,
  },
  feedbackText: {
    fontSize: 15,
    color: "#fff",
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
    color: "#ff80ab",
    textTransform: "uppercase",
  },
  probBarBg: {
    flex: 1,
    height: 32,
    backgroundColor: "rgba(255, 23, 68, 0.1)",
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
    color: "#fff",
    textAlign: "right",
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.9)",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  modal: {
    backgroundColor: "#1a0a14",
    borderRadius: 24,
    padding: 24,
    width: "100%",
    maxWidth: 500,
    borderWidth: 2,
    borderColor: "rgba(255, 23, 68, 0.3)",
  },
  modalClose: {
    position: "absolute",
    top: 16,
    right: 16,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(255, 23, 68, 0.2)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalCloseText: {
    color: "#fff",
    fontSize: 24,
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
    color: "#fff",
  },
  modalBody: {
    gap: 20,
  },
  modalSection: {},
  modalSectionTitle: {
    fontSize: 12,
    fontWeight: "800",
    color: "#ff4081",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 8,
  },
  modalSectionText: {
    fontSize: 15,
    color: "#ccc",
    lineHeight: 22,
  },
});
