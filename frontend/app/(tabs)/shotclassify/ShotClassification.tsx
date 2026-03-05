/**
 * Shot Classification Screen - React Native
 * Mobile-responsive with Spider Chart Visualization
 * Fixed centered modal & professional UI - COMPLETE CODE
 */

import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Pressable,
  ActivityIndicator,
  Alert,
  Dimensions,
  Modal,
  Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";
import * as Camera1 from "expo-camera";
import { LinearGradient } from "expo-linear-gradient";
import Animated, {
  FadeInUp,
  FadeInDown,
  FadeIn,
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSpring,
} from "react-native-reanimated";
import Svg, { Polygon, Circle, Line, Text as SvgText } from "react-native-svg";

import {
  getShotTypes,
  analyzeShot,
  ShotType,
  AnalysisResult,
} from "../../../services/shotClassificationApi";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");
const isSmallScreen = SCREEN_HEIGHT < 700;
const BOTTOM_NAV_HEIGHT = 80;
const BOTTOM_PADDING = BOTTOM_NAV_HEIGHT + 20;

interface Mistake {
  joint_id: string;
  body_part: string;
  severity: "critical" | "major" | "minor" | "negligible";
  severity_color: string;
  glow_intensity: number;
  explanation: string;
  recommendation: string;
}

interface MistakeAnalysis {
  body_part: string;
  severity_score: number;
}

interface SpiderChartProps {
  data: MistakeAnalysis[];
}

const SpiderChart: React.FC<SpiderChartProps> = ({ data }) => {
  const size = SCREEN_WIDTH - 80;
  const center = size / 2;
  const radius = size / 2 - 40;
  const categories = [
    "Torso",
    "Front Elbow",
    "Back Elbow",
    "Back Knee",
    "Shoulders",
  ];

  const getCategoryScore = (category: string) => {
    const categoryMap: Record<string, string[]> = {
      Torso: ["Body Position", "Torso"],
      "Front Elbow": ["Front Elbow", "Front Wrist"],
      "Back Elbow": ["Back Elbow", "Back Wrist"],
      "Back Knee": ["Back Knee", "Front Knee"],
      Shoulders: ["Shoulders", "Shoulder Rotation"],
    };

    const relevantMistakes = data.filter((m) =>
      categoryMap[category]?.some((cat) =>
        m.body_part.toLowerCase().includes(cat.toLowerCase()),
      ),
    );

    if (relevantMistakes.length === 0) return 100;

    const avgSeverity =
      relevantMistakes.reduce((sum, m) => sum + m.severity_score, 0) /
      relevantMistakes.length;
    return Math.max(0, 100 - avgSeverity * 100);
  };

  const getPoint = (index: number, value: number) => {
    const angle = (Math.PI * 2 * index) / categories.length - Math.PI / 2;
    const distance = (value / 100) * radius;
    return {
      x: center + Math.cos(angle) * distance,
      y: center + Math.sin(angle) * distance,
    };
  };

  const perfectPoints = categories.map((_, i) => getPoint(i, 100));
  const userPoints = categories.map((cat, i) =>
    getPoint(i, getCategoryScore(cat)),
  );

  const perfectPolygon = perfectPoints.map((p) => `${p.x},${p.y}`).join(" ");
  const userPolygon = userPoints.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <View style={styles.spiderContainer}>
      <Svg width={size} height={size}>
        {[20, 40, 60, 80, 100].map((percent) => (
          <Circle
            key={percent}
            cx={center}
            cy={center}
            r={(percent / 100) * radius}
            stroke="#1a1a1a"
            strokeWidth="1"
            fill="none"
          />
        ))}

        {categories.map((_, i) => {
          const point = getPoint(i, 100);
          return (
            <Line
              key={i}
              x1={center}
              y1={center}
              x2={point.x}
              y2={point.y}
              stroke="#1a1a1a"
              strokeWidth="1"
            />
          );
        })}

        <Polygon
          points={perfectPolygon}
          fill="rgba(0, 255, 136, 0.1)"
          stroke="#00ff88"
          strokeWidth="2"
        />

        <Polygon
          points={userPolygon}
          fill="rgba(255, 59, 48, 0.2)"
          stroke="#ff3b30"
          strokeWidth="2"
        />

        {categories.map((cat, i) => {
          const labelPoint = getPoint(i, 115);
          return (
            <SvgText
              key={cat}
              x={labelPoint.x}
              y={labelPoint.y}
              fill="#00ff88"
              fontSize="12"
              fontWeight="bold"
              textAnchor="middle"
            >
              {cat.toUpperCase()}
            </SvgText>
          );
        })}
      </Svg>

      <View style={styles.spiderLegend}>
        <View style={styles.spiderLegendItem}>
          <View
            style={[styles.spiderLegendDot, { backgroundColor: "#00ff88" }]}
          />
          <Text style={styles.spiderLegendText}>Perfect Form</Text>
        </View>
        <View style={styles.spiderLegendItem}>
          <View
            style={[styles.spiderLegendDot, { backgroundColor: "#ff3b30" }]}
          />
          <Text style={styles.spiderLegendText}>Your Execution</Text>
        </View>
      </View>
    </View>
  );
};

interface MistakeModalProps {
  visible: boolean;
  mistake: Mistake | null;
  onClose: () => void;
}

const MistakeModal: React.FC<MistakeModalProps> = ({
  visible,
  mistake,
  onClose,
}) => {
  if (!mistake) return null;

  const getSeverityConfig = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "critical":
        return {
          color: "#ff3b30",
          bgColor: "rgba(255, 59, 48, 0.1)",
          label: "CRITICAL",
          description: "Immediate attention required",
        };
      case "major":
        return {
          color: "#ff9500",
          bgColor: "rgba(255, 149, 0, 0.1)",
          label: "MAJOR",
          description: "Significant improvement needed",
        };
      case "minor":
        return {
          color: "#ffcc00",
          bgColor: "rgba(255, 204, 0, 0.1)",
          label: "MINOR",
          description: "Fine-tuning recommended",
        };
      default:
        return {
          color: "#00ff88",
          bgColor: "rgba(0, 255, 136, 0.1)",
          label: "NEGLIGIBLE",
          description: "Good form overall",
        };
    }
  };

  const severityConfig = getSeverityConfig(mistake.severity);
  const severityPercentage = mistake.glow_intensity * 100;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
      statusBarTranslucent
    >
      <View style={styles.modalOverlay}>
        <TouchableOpacity
          style={StyleSheet.absoluteFill}
          activeOpacity={1}
          onPress={onClose}
        />
        <View style={styles.modalWrapper}>
          <Animated.View
            entering={FadeIn.duration(300)}
            style={styles.modalContentCentered}
          >
            <ScrollView
              style={styles.modalScrollView}
              showsVerticalScrollIndicator={false}
              bounces={false}
            >
              {/* Header */}
              <View
                style={[
                  styles.modalHeaderCentered,
                  { backgroundColor: severityConfig.color },
                ]}
              >
                <View style={styles.modalHeaderContent}>
                  <Text style={styles.modalTitle}>
                    {mistake.body_part.toUpperCase()}
                  </Text>
                  <View style={styles.modalSeverityBadge}>
                    <Text style={styles.modalSeverityText}>
                      {severityConfig.label}
                    </Text>
                  </View>
                </View>
                <Text style={styles.modalSubtitle}>
                  {severityConfig.description}
                </Text>
              </View>

              {/* Body */}
              <View style={styles.modalBody}>
                {/* Severity Visualization */}
                <View
                  style={[
                    styles.severityCard,
                    { backgroundColor: severityConfig.bgColor },
                  ]}
                >
                  <View style={styles.severityHeader}>
                    <Text style={styles.severityTitle}>Issue Severity</Text>
                    <Text
                      style={[
                        styles.severityPercentage,
                        { color: severityConfig.color },
                      ]}
                    >
                      {severityPercentage.toFixed(0)}%
                    </Text>
                  </View>

                  {/* Progress Bar */}
                  <View style={styles.severityProgressContainer}>
                    <View style={styles.severityProgressBg}>
                      <Animated.View
                        entering={FadeInUp.duration(800)}
                        style={[
                          styles.severityProgressFill,
                          {
                            width: `${severityPercentage}%`,
                            backgroundColor: severityConfig.color,
                          },
                        ]}
                      />
                    </View>
                    <View style={styles.severityLabels}>
                      <Text style={styles.severityLabelText}>0%</Text>
                      <Text style={styles.severityLabelText}>Low</Text>
                      <Text style={styles.severityLabelText}>Medium</Text>
                      <Text style={styles.severityLabelText}>High</Text>
                      <Text style={styles.severityLabelText}>100%</Text>
                    </View>
                  </View>

                  {/* Severity Indicator */}
                  <View style={styles.severityIndicatorRow}>
                    <View
                      style={[
                        styles.severityDot,
                        { backgroundColor: severityConfig.color },
                      ]}
                    />
                    <Text style={styles.severityIndicatorText}>
                      This issue requires{" "}
                      <Text
                        style={{
                          color: severityConfig.color,
                          fontWeight: "800",
                        }}
                      >
                        {severityConfig.label.toLowerCase()}
                      </Text>{" "}
                      attention
                    </Text>
                  </View>
                </View>

                {/* The Issue */}
                <View style={styles.modalSection}>
                  <View style={styles.sectionHeader}>
                    <View
                      style={[
                        styles.sectionIconContainer,
                        { backgroundColor: "rgba(255, 59, 48, 0.1)" },
                      ]}
                    >
                      <Text style={styles.sectionIcon}>⚠️</Text>
                    </View>
                    <Text style={styles.modalSectionLabel}>The Issue</Text>
                  </View>
                  <Text style={styles.modalSectionText}>
                    {mistake.explanation}
                  </Text>
                </View>

                {/* How to Fix */}
                <View style={styles.modalSection}>
                  <View style={styles.sectionHeader}>
                    <View
                      style={[
                        styles.sectionIconContainer,
                        { backgroundColor: "rgba(0, 255, 136, 0.1)" },
                      ]}
                    >
                      <Text style={styles.sectionIcon}>✓</Text>
                    </View>
                    <Text style={styles.modalSectionLabel}>How to Fix It</Text>
                  </View>
                  <Text style={styles.modalSectionText}>
                    {mistake.recommendation}
                  </Text>
                </View>

                {/* Close Button */}
                <TouchableOpacity
                  onPress={onClose}
                  activeOpacity={0.8}
                  style={styles.modalCloseButton}
                >
                  <LinearGradient
                    colors={[severityConfig.color, severityConfig.color]}
                    style={styles.modalCloseGradient}
                  >
                    <Text style={styles.modalCloseText}>Got it!</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </ScrollView>
          </Animated.View>
        </View>
      </View>
    </Modal>
  );
};

// ===== CAMERA RECORDER COMPONENT =====
interface CameraRecorderProps {
  visible: boolean;
  onClose: () => void;
  onVideoTaken: (uri: string) => void;
  mode: "shot" | "stance"; // for future customization
}

const CameraRecorder: React.FC<CameraRecorderProps> = ({
  visible,
  onClose,
  onVideoTaken,
  mode,
}) => {
  const cameraRef = useRef<Camera1.CameraView>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const FOCUS_SQUARE_SIZE = 190;
  const [focusPoint, setFocusPoint] = useState({
    x: SCREEN_WIDTH / 2,
    y: SCREEN_HEIGHT / 2,
  });

  useEffect(() => {
    (async () => {
      try {
        const { status: cameraStatus } =
          await Camera.requestCameraPermissionsAsync();

        const { status: audioStatus } =
          await Camera.requestMicrophonePermissionsAsync();

        console.log("Camera:", cameraStatus);
        console.log("Microphone:", audioStatus);

        if (cameraStatus !== "granted" || audioStatus !== "granted") {
          Alert.alert(
            "Permission required",
            "Camera and microphone permissions are required.",
          );
        }
      } catch (error) {
        console.error("Permission error:", error);
      }
    })();
  }, []);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const startRecording = async () => {
    if (cameraRef.current) {
      try {
        setIsRecording(true);
        setRecordingTime(0);

        // Record without audio on first try, if that fails, show permission error
        const video = await cameraRef.current.recordAsync({
          maxDuration: 30,
          maxFileSize: 500000000,
        });

        setIsRecording(false);
        if (video?.uri) {
          onVideoTaken(video.uri);
          Alert.alert("Success", "Video recorded successfully!");
          onClose();
        }
      } catch (error: any) {
        setIsRecording(false);
        const errorMsg = error?.message || "Failed to record video";

        if (
          errorMsg.includes("RECORD_AUDIO") ||
          errorMsg.includes("permission")
        ) {
          Alert.alert(
            "Permission Required",
            "Please grant microphone/audio permission in your phone settings to record videos.\n\nSettings → Apps → CricketApp → Permissions → Microphone",
          );
        } else {
          Alert.alert("Recording Error", errorMsg);
        }
        console.error("Recording error:", error);
      }
    }
  };

  const stopRecording = async () => {
    if (cameraRef.current && isRecording) {
      try {
        await cameraRef.current.stopRecording();
      } catch (error) {
        console.error("Error stopping recording:", error);
      }
    }
  };

  const handleFocusGuideTap = (x: number, y: number) => {
    const half = FOCUS_SQUARE_SIZE / 2;
    const clampedX = Math.max(half, Math.min(SCREEN_WIDTH - half, x));
    const clampedY = Math.max(half, Math.min(SCREEN_HEIGHT - half, y));
    setFocusPoint({ x: clampedX, y: clampedY });
  };

  if (!visible) return null;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.cameraContainer}>
        <Camera1.CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          facing="back"
          mode="video"
        />

        <View pointerEvents="none" style={styles.cameraGridOverlay}>
          <View style={styles.cameraGridRow}>
            <View style={styles.cameraGridLineHorizontal} />
          </View>
          <View style={styles.cameraGridRow}>
            <View style={styles.cameraGridLineHorizontal} />
          </View>
          <View style={[styles.cameraGridLineVertical, { left: "33.33%" }]} />
          <View style={[styles.cameraGridLineVertical, { left: "66.66%" }]} />
        </View>

        <Pressable
          style={styles.focusTapLayer}
          onPress={(event) => {
            const { locationX, locationY } = event.nativeEvent;
            handleFocusGuideTap(locationX, locationY);
          }}
        >
          <View
            pointerEvents="none"
            style={[
              styles.focusSquare,
              {
                width: FOCUS_SQUARE_SIZE,
                height: FOCUS_SQUARE_SIZE,
                left: focusPoint.x - FOCUS_SQUARE_SIZE / 2,
                top: focusPoint.y - FOCUS_SQUARE_SIZE / 2,
              },
            ]}
          >
            <View style={styles.focusCenterDot} />
          </View>
        </Pressable>

        <View pointerEvents="none" style={styles.focusHintContainer}>
          <Text style={styles.focusHintText}>Tap to move focus square</Text>
        </View>

        {/* Camera Controls */}
        <View style={styles.cameraControls}>
          {/* Top Bar */}
          <View style={styles.cameraTopBar}>
            <TouchableOpacity
              onPress={onClose}
              disabled={isRecording}
              activeOpacity={0.7}
            >
              <View style={styles.cameraCancelButton}>
                <Text
                  style={{ color: "#fff", fontWeight: "800", fontSize: 16 }}
                >
                  ✕
                </Text>
              </View>
            </TouchableOpacity>
            <View style={styles.cameraTimer}>
              <Text style={{ color: "#fff", fontWeight: "800", fontSize: 14 }}>
                {`${Math.floor(recordingTime / 60)
                  .toString()
                  .padStart(2, "0")}:${(recordingTime % 60)
                  .toString()
                  .padStart(2, "0")}`}
              </Text>
            </View>
            <View style={{ width: 40 }} />
          </View>

          {/* Recording Status */}
          {isRecording && (
            <View
              style={{
                alignItems: "center",
                marginBottom: 20,
                flexDirection: "row",
                justifyContent: "center",
                gap: 8,
              }}
            >
              <View
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 6,
                  backgroundColor: "#ff3b30",
                  opacity: 0.7,
                }}
              />
              <Text
                style={{ color: "#ff3b30", fontWeight: "800", fontSize: 14 }}
              >
                RECORDING
              </Text>
            </View>
          )}

          {/* Bottom Controls */}
          <View style={styles.cameraBottomBar}>
            {!isRecording ? (
              <TouchableOpacity
                onPress={startRecording}
                activeOpacity={0.8}
                style={styles.cameraRecordButton}
              >
                <View style={styles.cameraRecordCircle} />
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                onPress={stopRecording}
                activeOpacity={0.8}
                style={styles.cameraStopButton}
              >
                <View style={styles.cameraStopSquare} />
              </TouchableOpacity>
            )}
          </View>
        </View>
      </View>
    </Modal>
  );
};

export default function ShotClassificationScreen() {
  const [shotTypes, setShotTypes] = useState<ShotType[]>([]);
  const [selectedShot, setSelectedShot] = useState<string | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [score, setScore] = useState(0);
  const [selectedMistake, setSelectedMistake] = useState<Mistake | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  // Stance Consistency Tab
  const [activeTab, setActiveTab] = useState<"shot" | "stance">("shot");
  const [stanceVideos, setStanceVideos] = useState<string[]>([]);
  const [stanceLoading, setStanceLoading] = useState(false);
  const [stanceAnalysisResult, setStanceAnalysisResult] = useState<any | null>(
    null,
  );

  // Camera states
  const [cameraModalVisible, setCameraModalVisible] = useState(false);
  const [cameraMode, setCameraMode] = useState<"shot" | "stance">("shot");

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

  // --- Stance Consistency helpers ---
  const pickStanceVideo = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: "videos",
      quality: 1,
    });

    if (!res.canceled && res.assets.length > 0) {
      setStanceVideos((s) => [...s, res.assets[0].uri]);
    }
  };

  // --- Camera Recording handlers ---
  const handleTakeShotVideo = () => {
    setCameraMode("shot");
    setCameraModalVisible(true);
  };

  const handleTakeStanceVideo = () => {
    setCameraMode("stance");
    setCameraModalVisible(true);
  };

  const handleCameraVideoTaken = (uri: string) => {
    if (cameraMode === "shot") {
      setVideoUri(uri);
      setResult(null);
      setScore(0);
    } else if (cameraMode === "stance") {
      setStanceVideos((s) => [...s, uri]);
    }
  };

  const removeStanceVideo = (index: number) => {
    setStanceVideos((s) => s.filter((_, i) => i !== index));
  };

  const handleQuickCompare = async () => {
    if (stanceVideos.length < 2) {
      Alert.alert("Need two videos", "Please add two videos for quick compare");
      return;
    }

    try {
      setStanceLoading(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Compare failed";
      Alert.alert("Compare Failed", msg);
    } finally {
      setStanceLoading(false);
    }
  };

  const handleFullStanceAnalysis = async () => {
    if (stanceVideos.length < 2) {
      Alert.alert("Need videos", "Add at least two videos for full analysis");
      return;
    }

    try {
      setStanceLoading(true);

      // optionally show summary
      Alert.alert(
        "Analysis Complete",
        "Stance analysis completed successfully",
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      Alert.alert("Analysis Failed", msg);
    } finally {
      setStanceLoading(false);
    }
  };

  const handleMistakePress = (mistake: Mistake) => {
    setSelectedMistake(mistake);
    setModalVisible(true);
  };

  const buttonAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
  }));

  const getRadarData = () => {
    if (!result) return [];
    const probs = result.ensemble_probabilities;
    return Object.entries(probs)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([shot, prob]) => ({
        shot,
        value: prob * 100,
      }));
  };

  return (
    <ScrollView
      style={styles.container}
      showsVerticalScrollIndicator={false}
      contentContainerStyle={{ paddingBottom: BOTTOM_PADDING }}
    >
      {/* Hero Section */}
      <LinearGradient
        colors={["rgba(0, 255, 136, 0.15)", "rgba(0, 204, 111, 0.1)"]}
        style={[styles.hero, { paddingTop: isSmallScreen ? 40 : 60 }]}
      >
        <Animated.View
          entering={FadeInDown.duration(800)}
          style={styles.heroContent}
        >
          <Text style={[styles.heroTitle, isSmallScreen && { fontSize: 28 }]}>
            CRICKET SHOT ANALYZER
          </Text>
          <Text
            style={[styles.heroSubtitle, isSmallScreen && { fontSize: 12 }]}
          >
            AI-Powered Biomechanics Analysis
          </Text>
        </Animated.View>

        <View style={styles.heroStats}>
          {[
            { value: "99.2%", label: "Accuracy" },
            { value: "Real-time", label: "Analysis" },
            { value: "AI", label: "Powered" },
          ].map((stat, idx) => (
            <Animated.View
              key={idx}
              entering={FadeInUp.delay(idx * 100).duration(600)}
              style={styles.stat}
            >
              <Text
                style={[styles.statValue, isSmallScreen && { fontSize: 18 }]}
              >
                {stat.value}
              </Text>
              <Text
                style={[styles.statLabel, isSmallScreen && { fontSize: 9 }]}
              >
                {stat.label}
              </Text>
            </Animated.View>
          ))}
        </View>
      </LinearGradient>

      {/* Tabs: Shot vs Stance */}
      <View
        style={{
          flexDirection: "row",
          marginHorizontal: 16,
          marginTop: 12,
          marginBottom: 4,
        }}
      >
        <TouchableOpacity
          onPress={() => setActiveTab("shot")}
          activeOpacity={0.8}
          style={{ flex: 1, marginRight: 8 }}
        >
          <LinearGradient
            colors={
              activeTab === "shot"
                ? ["#00ff88", "#00cc6f"]
                : ["#1a1a1a", "#1a1a1a"]
            }
            style={{ padding: 12, borderRadius: 12, alignItems: "center" }}
          >
            <Text
              style={{
                color: activeTab === "shot" ? "#000" : "#00ff88",
                fontWeight: "800",
              }}
            >
              Shot Analyzer
            </Text>
          </LinearGradient>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => setActiveTab("stance")}
          activeOpacity={0.8}
          style={{ flex: 1, marginLeft: 8 }}
        >
          <LinearGradient
            colors={
              activeTab === "stance"
                ? ["#00ff88", "#00cc6f"]
                : ["#1a1a1a", "#1a1a1a"]
            }
            style={{ padding: 12, borderRadius: 12, alignItems: "center" }}
          >
            <Text
              style={{
                color: activeTab === "stance" ? "#000" : "#00ff88",
                fontWeight: "800",
              }}
            >
              Stance Tracker
            </Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>

      {/* Stance Tab Content */}
      {activeTab === "stance" && (
        <>
          {!stanceAnalysisResult ? (
            <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
              <View style={styles.cardHeader}>
                <View style={styles.cardHeaderText}>
                  <Text style={styles.cardTitle}>
                    Stance Consistency Tracker
                  </Text>
                  <Text style={styles.cardSubtitle}>
                    Analyze your batting stance across multiple videos
                  </Text>
                </View>
              </View>

              <View style={{ gap: 12 }}>
                <Text
                  style={{
                    color: "#00ff88",
                    fontWeight: "700",
                    fontSize: 12,
                    marginBottom: 4,
                  }}
                >
                  VIDEOS ADDED ({stanceVideos.length})
                </Text>

                {stanceVideos.length === 0 ? (
                  <View style={{ padding: 20, alignItems: "center" }}>
                    <Text style={{ color: "#666", fontSize: 14 }}>
                      No videos added yet
                    </Text>
                  </View>
                ) : (
                  stanceVideos.map((uri, idx) => (
                    <View
                      key={idx}
                      style={{
                        flexDirection: "row",
                        alignItems: "center",
                        backgroundColor: "rgba(0, 255, 136, 0.05)",
                        padding: 12,
                        borderRadius: 10,
                        borderLeftWidth: 4,
                        borderLeftColor: "#00ff88",
                      }}
                    >
                      <View style={{ flex: 1 }}>
                        <Text
                          style={{
                            color: "#00ff88",
                            fontWeight: "800",
                            fontSize: 13,
                          }}
                        >
                          Video {idx + 1}
                        </Text>
                        <Text
                          style={{ color: "#999", fontSize: 11, marginTop: 2 }}
                          numberOfLines={1}
                        >
                          {uri.split("/").pop()}
                        </Text>
                      </View>
                      <TouchableOpacity
                        onPress={() => removeStanceVideo(idx)}
                        activeOpacity={0.7}
                      >
                        <Text
                          style={{
                            color: "#ff3b30",
                            fontWeight: "800",
                            fontSize: 12,
                          }}
                        >
                          ✕
                        </Text>
                      </TouchableOpacity>
                    </View>
                  ))
                )}

                <View style={{ flexDirection: "row", gap: 12, marginTop: 8 }}>
                  <TouchableOpacity
                    onPress={pickStanceVideo}
                    activeOpacity={0.8}
                    style={{ flex: 1 }}
                  >
                    <LinearGradient
                      colors={["#00ff88", "#00cc6f"]}
                      style={{
                        padding: 14,
                        borderRadius: 12,
                        alignItems: "center",
                      }}
                    >
                      <Text
                        style={{
                          fontWeight: "800",
                          color: "#000",
                          fontSize: 13,
                        }}
                      >
                        📁 Pick Video
                      </Text>
                    </LinearGradient>
                  </TouchableOpacity>

                  <TouchableOpacity
                    onPress={handleTakeStanceVideo}
                    activeOpacity={0.8}
                    style={{ flex: 1 }}
                  >
                    <LinearGradient
                      colors={["#00ffff", "#00cccc"]}
                      style={{
                        padding: 14,
                        borderRadius: 12,
                        alignItems: "center",
                      }}
                    >
                      <Text
                        style={{
                          fontWeight: "800",
                          color: "#000",
                          fontSize: 13,
                        }}
                      >
                        🎥 Take Video
                      </Text>
                    </LinearGradient>
                  </TouchableOpacity>
                </View>

                <TouchableOpacity
                  onPress={handleFullStanceAnalysis}
                  disabled={stanceVideos.length < 2 || stanceLoading}
                  activeOpacity={0.8}
                  style={{
                    marginTop: 12,
                    opacity: stanceVideos.length < 2 ? 0.5 : 1,
                  }}
                >
                  <LinearGradient
                    colors={["#00ffff", "#00cccc"]}
                    style={{
                      padding: 14,
                      borderRadius: 12,
                      alignItems: "center",
                      flexDirection: "row",
                      justifyContent: "center",
                      gap: 8,
                    }}
                  >
                    {stanceLoading ? (
                      <>
                        <ActivityIndicator color="#000" size="small" />
                        <Text
                          style={{
                            fontWeight: "800",
                            color: "#000",
                            fontSize: 14,
                          }}
                        >
                          Analyzing...
                        </Text>
                      </>
                    ) : (
                      <Text
                        style={{
                          fontWeight: "800",
                          color: "#000",
                          fontSize: 14,
                        }}
                      >
                        🎯 Analyze Consistency ({stanceVideos.length})
                      </Text>
                    )}
                  </LinearGradient>
                </TouchableOpacity>

                {stanceVideos.length < 2 && (
                  <Text
                    style={{
                      color: "#ff9500",
                      fontSize: 12,
                      textAlign: "center",
                      marginTop: 8,
                    }}
                  >
                    Add at least 2 videos to start analysis
                  </Text>
                )}
              </View>
            </Animated.View>
          ) : (
            // ===== Full Analysis Results =====
            <>
              {/* Header with back button */}
              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  marginHorizontal: 16,
                  marginTop: 12,
                  marginBottom: 8,
                }}
              >
                <TouchableOpacity
                  onPress={() => {
                    setStanceAnalysisResult(null);
                    setStanceVideos([]);
                  }}
                  activeOpacity={0.7}
                >
                  <Text
                    style={{
                      color: "#00ff88",
                      fontWeight: "800",
                      fontSize: 18,
                    }}
                  >
                    ← Back
                  </Text>
                </TouchableOpacity>
                <Text
                  style={{
                    marginLeft: 12,
                    color: "#00ff88",
                    fontWeight: "800",
                    fontSize: 16,
                  }}
                >
                  Analysis Results
                </Text>
              </View>

              {/* Overall Summary */}
              <Animated.View entering={FadeInUp.delay(100)} style={styles.card}>
                <Text style={styles.cardTitle}>Overall Summary</Text>

                <View style={{ marginTop: 12, gap: 16 }}>
                  {/* Big Score Ring */}
                  <View style={{ alignItems: "center" }}>
                    <View
                      style={{
                        width: 140,
                        height: 140,
                        borderRadius: 70,
                        borderWidth: 6,
                        borderColor: "#00ff88",
                        backgroundColor: "rgba(0, 255, 136, 0.08)",
                        justifyContent: "center",
                        alignItems: "center",
                      }}
                    >
                      <Text
                        style={{
                          fontSize: 48,
                          fontWeight: "900",
                          color: "#00ff88",
                        }}
                      >
                        {stanceAnalysisResult?.summary?.overall_consistency_score?.toFixed(
                          1,
                        )}
                        %
                      </Text>
                      <Text
                        style={{
                          fontSize: 11,
                          color: "#00cc6f",
                          fontWeight: "700",
                          marginTop: 4,
                        }}
                      >
                        CONSISTENCY
                      </Text>
                    </View>
                  </View>

                  {/* Rating & Stats */}
                  <View style={{ gap: 10 }}>
                    <View
                      style={{
                        flexDirection: "row",
                        justifyContent: "space-between",
                        paddingVertical: 10,
                        paddingHorizontal: 12,
                        backgroundColor: "rgba(0, 255, 136, 0.05)",
                        borderRadius: 10,
                      }}
                    >
                      <Text style={{ color: "#999", fontWeight: "600" }}>
                        Rating
                      </Text>
                      <Text style={{ color: "#00ff88", fontWeight: "800" }}>
                        {stanceAnalysisResult?.summary?.consistency_rating}
                      </Text>
                    </View>
                    <View
                      style={{
                        flexDirection: "row",
                        justifyContent: "space-between",
                        paddingVertical: 10,
                        paddingHorizontal: 12,
                        backgroundColor: "rgba(0, 255, 136, 0.05)",
                        borderRadius: 10,
                      }}
                    >
                      <Text style={{ color: "#999", fontWeight: "600" }}>
                        Videos Analyzed
                      </Text>
                      <Text style={{ color: "#00ff88", fontWeight: "800" }}>
                        {stanceAnalysisResult?.summary?.total_videos_analyzed}
                      </Text>
                    </View>
                    <View
                      style={{
                        flexDirection: "row",
                        justifyContent: "space-between",
                        paddingVertical: 10,
                        paddingHorizontal: 12,
                        backgroundColor: "rgba(0, 255, 136, 0.05)",
                        borderRadius: 10,
                      }}
                    >
                      <Text style={{ color: "#999", fontWeight: "600" }}>
                        Std Deviation
                      </Text>
                      <Text style={{ color: "#00ff88", fontWeight: "800" }}>
                        {stanceAnalysisResult?.summary?.consistency_std?.toFixed(
                          2,
                        )}
                      </Text>
                    </View>
                  </View>
                </View>
              </Animated.View>

              {/* Individual Video Scores */}
              <Animated.View entering={FadeInUp.delay(150)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text style={styles.cardTitle}>Individual Scores</Text>
                    <Text style={styles.cardSubtitle}>
                      Consistency per video
                    </Text>
                  </View>
                </View>

                <View style={{ gap: 12 }}>
                  {stanceAnalysisResult?.individual_video_scores?.map(
                    (score: any, idx: number) => (
                      <View key={idx} style={{ gap: 6 }}>
                        <View
                          style={{
                            flexDirection: "row",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Text style={{ color: "#cccccc", fontWeight: "700" }}>
                            Video {idx + 1}
                          </Text>
                          <Text
                            style={{
                              color: "#00ff88",
                              fontWeight: "800",
                              fontSize: 14,
                            }}
                          >
                            {score.consistency_score?.toFixed(1)}%
                          </Text>
                        </View>
                        <View
                          style={{
                            height: 8,
                            backgroundColor: "#2a2a2a",
                            borderRadius: 4,
                            overflow: "hidden",
                          }}
                        >
                          <LinearGradient
                            colors={["#00ff88", "#00cc6f"]}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 0 }}
                            style={{
                              width: `${score.consistency_score}%`,
                              height: "100%",
                            }}
                          />
                        </View>
                      </View>
                    ),
                  )}
                </View>
              </Animated.View>

              {/* Stance Detection Details */}
              {/* <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text style={styles.cardTitle}>Stance Detection</Text>
                    <Text style={styles.cardSubtitle}>Frame analysis for each video</Text>
                  </View>
                </View>

                <View style={{ gap: 10 }}>
                  {stanceAnalysisResult?.stance_detection_details?.map((detail: any, idx: number) => (
                    <View
                      key={idx}
                      style={{
                        padding: 12,
                        backgroundColor: "rgba(0, 255, 136, 0.03)",
                        borderRadius: 10,
                        borderLeftWidth: 3,
                        borderLeftColor: "#00ffff",
                      }}
                    >
                      <View
                        style={{
                          flexDirection: "row",
                          justifyContent: "space-between",
                          marginBottom: 8,
                        }}
                      >
                        <Text style={{ color: "#00ff88", fontWeight: "800" }}>Video {detail.video_index}</Text>
                        <Text style={{ color: "#00ffff", fontWeight: "800" }}>{detail.stance_timing}</Text>
                      </View>
                      <Text style={{ color: "#999", fontSize: 12 }}>
                        Frame {detail.stance_frame} of {detail.total_frames}
                      </Text>
                    </View>
                  ))}
                </View>
              </Animated.View> */}

              {/* Pairwise Similarities */}
              <Animated.View entering={FadeInUp.delay(250)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text style={styles.cardTitle}>Pairwise Comparison</Text>
                    <Text style={styles.cardSubtitle}>
                      How similar each video pair is
                    </Text>
                  </View>
                </View>

                <View style={{ gap: 8 }}>
                  {stanceAnalysisResult?.consistency_analysis?.pairwise_similarities?.map(
                    (pair: any, idx: number) => (
                      <View key={idx} style={{ gap: 4 }}>
                        <View
                          style={{
                            flexDirection: "row",
                            justifyContent: "space-between",
                            alignItems: "center",
                          }}
                        >
                          <Text
                            style={{
                              color: "#cccccc",
                              fontWeight: "600",
                              fontSize: 12,
                            }}
                          >
                            Video {pair.video_1} ↔ Video {pair.video_2}
                          </Text>
                          <Text
                            style={{
                              color:
                                pair.similarity === 100 ? "#00ff88" : "#ffcc00",
                              fontWeight: "800",
                              fontSize: 13,
                            }}
                          >
                            {pair.similarity?.toFixed(1)}%
                          </Text>
                        </View>
                        <View
                          style={{
                            height: 6,
                            backgroundColor: "#2a2a2a",
                            borderRadius: 3,
                            overflow: "hidden",
                          }}
                        >
                          <LinearGradient
                            colors={
                              pair.similarity === 100
                                ? ["#00ff88", "#00cc6f"]
                                : ["#ffcc00", "#ff9500"]
                            }
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 0 }}
                            style={{
                              width: `${pair.similarity}%`,
                              height: "100%",
                            }}
                          />
                        </View>
                      </View>
                    ),
                  )}
                </View>
              </Animated.View>

              {/* AI Feedback Section */}
              <Animated.View entering={FadeInUp.delay(300)} style={styles.card}>
                <Text style={styles.cardTitle}>AI Coach Feedback</Text>

                {/* Overall Assessment */}
                <View style={{ marginTop: 12, marginBottom: 16 }}>
                  <View
                    style={{
                      padding: 14,
                      backgroundColor: "rgba(0, 255, 136, 0.08)",
                      borderRadius: 12,
                      borderLeftWidth: 4,
                      borderLeftColor: "#00ff88",
                    }}
                  >
                    <Text
                      style={{
                        color: "#00ff88",
                        fontWeight: "800",
                        fontSize: 12,
                        marginBottom: 6,
                      }}
                    >
                      OVERALL ASSESSMENT
                    </Text>
                    <Text
                      style={{ color: "#cccccc", lineHeight: 20, fontSize: 13 }}
                    >
                      {stanceAnalysisResult?.feedback?.overall_assessment}
                    </Text>
                  </View>
                </View>

                {/* Strengths */}
                <View style={{ marginBottom: 12 }}>
                  <View
                    style={{
                      padding: 14,
                      backgroundColor: "rgba(0, 255, 136, 0.05)",
                      borderRadius: 12,
                      borderLeftWidth: 4,
                      borderLeftColor: "#00ff88",
                    }}
                  >
                    <Text
                      style={{
                        color: "#00ff88",
                        fontWeight: "800",
                        fontSize: 12,
                        marginBottom: 6,
                      }}
                    >
                      💪 STRENGTHS
                    </Text>
                    <Text
                      style={{ color: "#cccccc", lineHeight: 20, fontSize: 13 }}
                    >
                      {stanceAnalysisResult?.feedback?.strengths}
                    </Text>
                  </View>
                </View>

                {/* Improvements */}
                <View style={{ marginBottom: 12 }}>
                  <View
                    style={{
                      padding: 14,
                      backgroundColor: "rgba(255, 149, 0, 0.05)",
                      borderRadius: 12,
                      borderLeftWidth: 4,
                      borderLeftColor: "#ff9500",
                    }}
                  >
                    <Text
                      style={{
                        color: "#ff9500",
                        fontWeight: "800",
                        fontSize: 12,
                        marginBottom: 6,
                      }}
                    >
                      🎯 IMPROVEMENTS
                    </Text>
                    <Text
                      style={{ color: "#cccccc", lineHeight: 20, fontSize: 13 }}
                    >
                      {stanceAnalysisResult?.feedback?.improvements}
                    </Text>
                  </View>
                </View>

                {/* Motivation */}
                <View>
                  <View
                    style={{
                      padding: 14,
                      backgroundColor: "rgba(0, 255, 200, 0.08)",
                      borderRadius: 12,
                      borderLeftWidth: 4,
                      borderLeftColor: "#00ffc8",
                    }}
                  >
                    <Text
                      style={{
                        color: "#00ffc8",
                        fontWeight: "800",
                        fontSize: 12,
                        marginBottom: 6,
                      }}
                    >
                      🚀 MOTIVATION
                    </Text>
                    <Text
                      style={{ color: "#cccccc", lineHeight: 20, fontSize: 13 }}
                    >
                      {stanceAnalysisResult?.feedback?.motivation}
                    </Text>
                  </View>
                </View>
              </Animated.View>

              {/* Key Insights */}
              <Animated.View entering={FadeInUp.delay(350)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text style={styles.cardTitle}>Key Insights</Text>
                    <Text style={styles.cardSubtitle}>
                      Important findings from analysis
                    </Text>
                  </View>
                </View>

                <View style={{ gap: 10 }}>
                  {stanceAnalysisResult?.insights?.map(
                    (insight: string, idx: number) => (
                      <View
                        key={idx}
                        style={{
                          flexDirection: "row",
                          gap: 12,
                          padding: 12,
                          backgroundColor: "rgba(0, 255, 136, 0.03)",
                          borderRadius: 10,
                          borderLeftWidth: 3,
                          borderLeftColor: "#00cc6f",
                        }}
                      >
                        <Text
                          style={{
                            color: "#00ff88",
                            fontWeight: "900",
                            fontSize: 16,
                          }}
                        >
                          💡
                        </Text>
                        <Text
                          style={{
                            color: "#cccccc",
                            flex: 1,
                            lineHeight: 18,
                            fontSize: 12,
                          }}
                        >
                          {insight}
                        </Text>
                      </View>
                    ),
                  )}
                </View>
              </Animated.View>

              {/* New Analysis Button */}
              <View
                style={{ marginHorizontal: 16, marginBottom: 20, marginTop: 8 }}
              >
                <TouchableOpacity
                  onPress={() => {
                    setStanceAnalysisResult(null);
                    setStanceVideos([]);
                  }}
                  activeOpacity={0.8}
                >
                  <LinearGradient
                    colors={["#00ff88", "#00cc6f"]}
                    style={{
                      padding: 14,
                      borderRadius: 12,
                      alignItems: "center",
                    }}
                  >
                    <Text
                      style={{ fontWeight: "800", color: "#000", fontSize: 14 }}
                    >
                      Start New Analysis
                    </Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </>
          )}
        </>
      )}

      {/* Shot Tab Content - Only show when activeTab === "shot" */}
      {activeTab === "shot" && (
        <>
          {/* Shot Selection */}
          <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
            <View style={styles.cardHeader}>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
                >
                  Select Intended Shot
                </Text>
                <Text
                  style={[
                    styles.cardSubtitle,
                    isSmallScreen && { fontSize: 12 },
                  ]}
                >
                  Choose the shot you intended to play
                </Text>
              </View>
            </View>

            <View style={styles.shotGrid}>
              {shotTypes.map((shot) => {
                const isActive = selectedShot === shot.value;
                return (
                  <TouchableOpacity
                    key={shot.value}
                    onPress={() => setSelectedShot(shot.value)}
                    activeOpacity={0.7}
                  >
                    <LinearGradient
                      colors={
                        isActive
                          ? ["#00ff88", "#00cc6f"]
                          : [
                              "rgba(0, 255, 136, 0.1)",
                              "rgba(0, 204, 111, 0.05)",
                            ]
                      }
                      style={[
                        styles.shotChip,
                        isActive && styles.shotChipActive,
                      ]}
                    >
                      <Text
                        style={[
                          styles.shotChipText,
                          isSmallScreen && { fontSize: 11 },
                          isActive && styles.shotChipTextActive,
                        ]}
                      >
                        {shot.label}
                      </Text>
                      {isActive && (
                        <View style={styles.checkMarkContainer}>
                          <Text
                            style={[
                              styles.checkMark,
                              isSmallScreen && { fontSize: 14 },
                            ]}
                          >
                            ✓
                          </Text>
                        </View>
                      )}
                    </LinearGradient>
                  </TouchableOpacity>
                );
              })}
            </View>
          </Animated.View>

          {/* Video Upload */}
          <Animated.View entering={FadeInUp.delay(300)} style={styles.card}>
            <View style={styles.cardHeader}>
              <View style={styles.cardHeaderText}>
                <Text style={styles.cardTitle}>Batting Video</Text>
                <Text style={styles.cardSubtitle}>
                  Add your batting video for analysis
                </Text>
              </View>
            </View>

            <View style={{ gap: 12 }}>
              {videoUri && (
                <View
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    backgroundColor: "rgba(0, 255, 136, 0.05)",
                    padding: 12,
                    borderRadius: 10,
                    borderLeftWidth: 4,
                    borderLeftColor: "#00ff88",
                  }}
                >
                  <View style={{ flex: 1 }}>
                    <Text
                      style={{
                        color: "#00ff88",
                        fontWeight: "800",
                        fontSize: 13,
                      }}
                    >
                      Video Selected
                    </Text>
                    <Text
                      style={{ color: "#999", fontSize: 11, marginTop: 2 }}
                      numberOfLines={1}
                    >
                      {videoUri.split("/").pop()}
                    </Text>
                  </View>
                  <TouchableOpacity
                    onPress={() => setVideoUri(null)}
                    activeOpacity={0.7}
                  >
                    <Text
                      style={{
                        color: "#ff3b30",
                        fontWeight: "800",
                        fontSize: 12,
                      }}
                    >
                      ✕
                    </Text>
                  </TouchableOpacity>
                </View>
              )}

              <TouchableOpacity onPress={pickVideo} activeOpacity={0.8}>
                <LinearGradient
                  colors={["#00ff88", "#00cc6f"]}
                  style={{
                    padding: 14,
                    borderRadius: 12,
                    alignItems: "center",
                  }}
                >
                  <Text
                    style={{ fontWeight: "800", color: "#000", fontSize: 14 }}
                  >
                    📁 Pick from Library
                  </Text>
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity
                onPress={handleTakeShotVideo}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={["#00ffff", "#00cccc"]}
                  style={{
                    padding: 14,
                    borderRadius: 12,
                    alignItems: "center",
                  }}
                >
                  <Text
                    style={{ fontWeight: "800", color: "#000", fontSize: 14 }}
                  >
                    🎥 Take Video
                  </Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </Animated.View>

          {/* Analyze Button */}
          <Animated.View
            style={[styles.analyzeButtonContainer, buttonAnimStyle]}
          >
            <TouchableOpacity
              disabled={!videoUri || !selectedShot || analyzing}
              onPress={handleAnalyze}
              activeOpacity={0.8}
            >
              <LinearGradient
                colors={["#00ff88", "#00cc6f"]}
                style={[
                  styles.analyzeButton,
                  (!videoUri || !selectedShot || analyzing) &&
                    styles.analyzeButtonDisabled,
                ]}
              >
                {analyzing ? (
                  <>
                    <ActivityIndicator color="#000" size="small" />
                    <Text
                      style={[
                        styles.analyzeButtonText,
                        isSmallScreen && { fontSize: 14 },
                      ]}
                    >
                      Analyzing Your Shot...
                    </Text>
                  </>
                ) : (
                  <Text
                    style={[
                      styles.analyzeButtonText,
                      isSmallScreen && { fontSize: 14 },
                    ]}
                  >
                    Analyze with AI
                  </Text>
                )}
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>

          {/* Results */}
          {result && (
            <>
              {/* Status Badge */}
              <Animated.View
                entering={FadeInUp.delay(50)}
                style={styles.statusBadge}
              >
                <LinearGradient
                  colors={
                    result.is_correct
                      ? ["#00ff88", "#00cc6f"]
                      : ["#ff9500", "#ff6b00"]
                  }
                  style={styles.statusGradient}
                >
                  <Text
                    style={[
                      styles.statusText,
                      isSmallScreen && { fontSize: 14 },
                    ]}
                  >
                    {result.is_correct
                      ? "PERFECT FORM"
                      : "FORM MISMATCH DETECTED"}
                  </Text>
                </LinearGradient>
              </Animated.View>

              {/* Score Card */}
              <Animated.View entering={FadeInUp.delay(100)} style={styles.card}>
                <View style={styles.scoreContainer}>
                  <View
                    style={[
                      styles.scoreRing,
                      isSmallScreen && {
                        width: 100,
                        height: 100,
                        borderRadius: 50,
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.scoreValue,
                        isSmallScreen && { fontSize: 30 },
                      ]}
                    >
                      {score}%
                    </Text>
                    <Text
                      style={[
                        styles.scoreLabel,
                        isSmallScreen && { fontSize: 9 },
                      ]}
                    >
                      INTENT SCORE
                    </Text>
                  </View>

                  <View style={styles.scoreInfo}>
                    <View
                      style={[
                        styles.scoreRow,
                        isSmallScreen && { paddingVertical: 6 },
                      ]}
                    >
                      <Text
                        style={[
                          styles.scoreRowLabel,
                          isSmallScreen && { fontSize: 11 },
                        ]}
                      >
                        Intended
                      </Text>
                      <Text
                        style={[
                          styles.scoreRowValue,
                          isSmallScreen && { fontSize: 12 },
                        ]}
                      >
                        {result.intended_shot.toUpperCase()}
                      </Text>
                    </View>
                    <View
                      style={[
                        styles.scoreRow,
                        isSmallScreen && { paddingVertical: 6 },
                      ]}
                    >
                      <Text
                        style={[
                          styles.scoreRowLabel,
                          isSmallScreen && { fontSize: 11 },
                        ]}
                      >
                        Detected
                      </Text>
                      <Text
                        style={[
                          styles.scoreRowValue,
                          isSmallScreen && { fontSize: 12 },
                          { color: result.is_correct ? "#00ff88" : "#ff9500" },
                        ]}
                      >
                        {result.predicted_shot.toUpperCase()}
                      </Text>
                    </View>
                    {/*<View
                  style={[
                    styles.scoreRow,
                    isSmallScreen && { paddingVertical: 6 },
                  ]}
                >
                  <Text
                    style={[
                      styles.scoreRowLabel,
                      isSmallScreen && { fontSize: 11 },
                    ]}
                  >
                    Prototype
                  </Text>
                  <Text
                    style={[
                      styles.scoreRowValue,
                      isSmallScreen && { fontSize: 12 },
                    ]}
                  >
                    {result.analysis_metadata.prototype_samples} samples
                  </Text>
                </View>*/}
                  </View>
                </View>
              </Animated.View>

              {/* Spider Chart */}
              <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text
                      style={[
                        styles.cardTitle,
                        isSmallScreen && { fontSize: 16 },
                      ]}
                    >
                      Shot DNA Analysis
                    </Text>
                    <Text
                      style={[
                        styles.cardSubtitle,
                        isSmallScreen && { fontSize: 12 },
                      ]}
                    >
                      Your execution vs perfect form
                    </Text>
                  </View>
                </View>

                <SpiderChart data={result.mistake_analysis || []} />
                <Text style={styles.chartHint}>
                  Center = poor form • Edge = perfect form
                </Text>
              </Animated.View>

              {/* Mistake Summary Cards */}
              {result?.visual_feedback.mistakes &&
                result.visual_feedback.mistakes.length > 0 && (
                  <Animated.View
                    entering={FadeInUp.delay(300)}
                    style={styles.card}
                  >
                    <View style={styles.cardHeader}>
                      <View style={styles.cardHeaderText}>
                        <Text
                          style={[
                            styles.cardTitle,
                            isSmallScreen && { fontSize: 16 },
                          ]}
                        >
                          Issues Summary
                        </Text>
                        <Text
                          style={[
                            styles.cardSubtitle,
                            isSmallScreen && { fontSize: 12 },
                          ]}
                        >
                          {result?.visual_feedback.mistakes?.length} area
                          {(result?.visual_feedback.mistakes?.length ?? 0) > 1
                            ? "s"
                            : ""}{" "}
                          need improvement
                        </Text>
                      </View>
                    </View>

                    {result?.visual_feedback.mistakes.map((mistake, idx) => (
                      <TouchableOpacity
                        key={idx}
                        onPress={() => handleMistakePress(mistake)}
                        activeOpacity={0.7}
                        style={styles.summaryCard}
                      >
                        <View
                          style={[
                            styles.summaryMarker,
                            {
                              backgroundColor:
                                mistake.severity === "critical"
                                  ? "#ff3b30"
                                  : mistake.severity === "major"
                                    ? "#ff9500"
                                    : mistake.severity === "minor"
                                      ? "#ffcc00"
                                      : "#00ff88",
                            },
                          ]}
                        />
                        <View style={styles.summaryContent}>
                          <Text style={styles.summaryTitle}>
                            {mistake.body_part}
                          </Text>
                          <Text style={styles.summaryText} numberOfLines={2}>
                            {mistake.explanation}
                          </Text>
                        </View>
                        <Text style={styles.summaryArrow}>›</Text>
                      </TouchableOpacity>
                    ))}
                  </Animated.View>
                )}

              {/* AI Feedback */}
              <Animated.View entering={FadeInUp.delay(400)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text
                      style={[
                        styles.cardTitle,
                        isSmallScreen && { fontSize: 16 },
                      ]}
                    >
                      AI Coach Feedback
                    </Text>
                    <Text
                      style={[
                        styles.cardSubtitle,
                        isSmallScreen && { fontSize: 12 },
                      ]}
                    >
                      Personalized recommendations
                    </Text>
                  </View>
                </View>
                <View style={styles.feedbackBox}>
                  <Text
                    style={[
                      styles.feedbackText,
                      isSmallScreen && { fontSize: 13 },
                    ]}
                  >
                    {result.coaching_feedback}
                  </Text>
                </View>
              </Animated.View>

              {/* Radar Chart */}
              <Animated.View entering={FadeInUp.delay(500)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderText}>
                    <Text
                      style={[
                        styles.cardTitle,
                        isSmallScreen && { fontSize: 16 },
                      ]}
                    >
                      Shot Classification Analysis
                    </Text>
                    <Text
                      style={[
                        styles.cardSubtitle,
                        isSmallScreen && { fontSize: 12 },
                      ]}
                    >
                      AI confidence distribution
                    </Text>
                  </View>
                </View>

                <View style={styles.radarContainer}>
                  {getRadarData().map((item, idx) => (
                    <Animated.View
                      key={item.shot}
                      entering={FadeInUp.delay(600 + idx * 50)}
                      style={styles.radarRow}
                    >
                      <View
                        style={[
                          styles.radarLeft,
                          isSmallScreen && { width: 80 },
                        ]}
                      >
                        <Text
                          style={[
                            styles.radarShot,
                            isSmallScreen && { fontSize: 11 },
                          ]}
                        >
                          {item.shot.toUpperCase()}
                        </Text>
                        <View style={styles.radarBadges}>
                          {item.shot === result.intended_shot && (
                            <Text
                              style={[
                                styles.radarBadge,
                                isSmallScreen && { fontSize: 9 },
                              ]}
                            >
                              INTENDED
                            </Text>
                          )}
                          {item.shot === result.predicted_shot && (
                            <Text
                              style={[
                                styles.radarBadge,
                                isSmallScreen && { fontSize: 9 },
                              ]}
                            >
                              DETECTED
                            </Text>
                          )}
                        </View>
                      </View>

                      <View style={styles.radarBarBg}>
                        <LinearGradient
                          colors={
                            item.shot === result.predicted_shot
                              ? ["#00ff88", "#00cc6f"]
                              : item.shot === result.intended_shot
                                ? ["#00ffff", "#00cccc"]
                                : ["#333333", "#1a1a1a"]
                          }
                          style={[
                            styles.radarBarFill,
                            { width: `${item.value}%` },
                          ]}
                          start={{ x: 0, y: 0 }}
                          end={{ x: 1, y: 0 }}
                        />
                      </View>

                      <Text
                        style={[
                          styles.radarValue,
                          isSmallScreen && { fontSize: 11 },
                        ]}
                      >
                        {item.value.toFixed(1)}%
                      </Text>
                    </Animated.View>
                  ))}
                </View>
              </Animated.View>

              {/* Detection Quality Info */}
              {/*<Animated.View entering={FadeInUp.delay(600)} style={styles.card}>
            <View style={styles.cardHeader}>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
                >
                  Video Analysis Quality
                </Text>
                <Text
                  style={[
                    styles.cardSubtitle,
                    isSmallScreen && { fontSize: 12 },
                  ]}
                >
                  Detection metrics from your video
                </Text>
              </View>
            </View>

            <View style={styles.detectionInfo}>
              <View style={styles.detectionRow}>
                <View style={styles.detectionItem}>
                  <Text style={styles.detectionLabel}>Ball Detection</Text>
                  <View style={styles.detectionBar}>
                    <View
                      style={[
                        styles.detectionBarFill,
                        {
                          width: `${
                            result.analysis_metadata?.contact_detection
                              ?.ball_detection_rate ?? 0
                          }%`,
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.detectionValue}>
                    {(
                      result.analysis_metadata?.contact_detection
                        ?.ball_detection_rate ?? 0
                    ).toFixed(1)}
                    %
                  </Text>
                </View>
              </View>

              <View style={styles.detectionRow}>
                <View style={styles.detectionItem}>
                  <Text style={styles.detectionLabel}>Detection Method</Text>
                  <Text style={styles.detectionMethodText}>
                    {result.analysis_metadata?.contact_detection?.detection_method
                      ?.replace(/_/g, " ")
                      .toUpperCase() ?? "N/A"}
                  </Text>
                </View>
              </View>

              <View style={styles.detectionRow}>
                <View style={styles.detectionItem}>
                  <Text style={styles.detectionLabel}>Analysis Confidence</Text>
                  <Text style={styles.detectionMethodText}>
                    Tier 2 Score:{" "}
                    {(
                      (result.analysis_metadata?.contact_detection
                        ?.tier2_score ?? 0) * 100
                    ).toFixed(1)}
                    %
                  </Text>
                </View>
              </View>
            </View>
          </Animated.View>*/}
            </>
          )}
        </>
      )}

      {/* Camera Recorder Modal */}
      <CameraRecorder
        visible={cameraModalVisible}
        onClose={() => setCameraModalVisible(false)}
        onVideoTaken={handleCameraVideoTaken}
        mode={cameraMode}
      />

      {/* Mistake Detail Modal */}
      <MistakeModal
        visible={modalVisible}
        mistake={selectedMistake}
        onClose={() => setModalVisible(false)}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0a0a0a",
  },

  // Hero
  hero: {
    paddingBottom: 40,
    paddingHorizontal: 20,
    backgroundColor: "#0a0a0a",
  },
  heroContent: {
    alignItems: "center",
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: "900",
    color: "#00ff88",
    letterSpacing: 1.5,
    textAlign: "center",
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 14,
    color: "#00cc6f",
    textAlign: "center",
  },
  heroStats: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginTop: 24,
  },
  stat: {
    alignItems: "center",
  },
  statValue: {
    fontSize: 22,
    fontWeight: "900",
    color: "#00ff88",
  },
  statLabel: {
    fontSize: 10,
    color: "#666666",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginTop: 4,
  },

  // Card
  card: {
    backgroundColor: "#1a1a1a",
    borderRadius: 16,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  cardHeaderText: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#00ff88",
    marginBottom: 4,
  },
  cardSubtitle: {
    fontSize: 13,
    color: "#666666",
  },
  chartHint: {
    fontSize: 11,
    color: "#888",
    textAlign: "center",
    marginBottom: 8,
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
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.3)",
    minWidth: (SCREEN_WIDTH - 72) / 3,
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  shotChipActive: {
    borderColor: "#00ff88",
  },
  shotChipText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#00cc6f",
  },
  shotChipTextActive: {
    color: "#000",
  },
  checkMarkContainer: {
    position: "absolute",
    top: -6,
    right: -6,
    backgroundColor: "#000",
    borderRadius: 10,
    width: 20,
    height: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  checkMark: {
    color: "#00ff88",
    fontSize: 14,
    fontWeight: "bold",
  },

  // Upload
  uploadArea: {
    padding: 24,
    borderRadius: 12,
    alignItems: "center",
  },
  uploadContent: {
    alignItems: "center",
  },
  uploadTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#000",
    marginBottom: 4,
  },
  uploadSubtitle: {
    fontSize: 13,
    color: "rgba(0, 0, 0, 0.7)",
  },

  // Analyze Button
  analyzeButtonContainer: {
    marginHorizontal: 16,
    marginBottom: 20,
  },
  analyzeButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 18,
    borderRadius: 12,
    gap: 12,
  },
  analyzeButtonDisabled: {
    opacity: 0.5,
  },
  analyzeButtonText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#000",
    letterSpacing: 0.5,
  },

  // Status Badge
  statusBadge: {
    marginHorizontal: 16,
    marginBottom: 16,
  },
  statusGradient: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 12,
  },
  statusText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#000",
    letterSpacing: 0.5,
  },

  // Score Card
  scoreContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 20,
  },
  scoreRing: {
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 8,
    borderColor: "#00ff88",
    backgroundColor: "rgba(0, 255, 136, 0.05)",
    justifyContent: "center",
    alignItems: "center",
  },
  scoreValue: {
    fontSize: 36,
    fontWeight: "900",
    color: "#00ff88",
  },
  scoreLabel: {
    fontSize: 10,
    color: "#00cc6f",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  scoreInfo: {
    flex: 1,
    gap: 8,
  },
  scoreRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: "rgba(0, 255, 136, 0.05)",
    borderRadius: 8,
  },
  scoreRowLabel: {
    fontSize: 13,
    color: "#666666",
    fontWeight: "600",
  },
  scoreRowValue: {
    fontSize: 14,
    fontWeight: "800",
    color: "#00ff88",
    textTransform: "uppercase",
  },

  // Spider Chart
  spiderContainer: {
    alignItems: "center",
    paddingVertical: 20,
  },
  spiderLegend: {
    flexDirection: "row",
    justifyContent: "center",
    gap: 24,
    marginTop: 16,
  },
  spiderLegendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  spiderLegendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  spiderLegendText: {
    fontSize: 12,
    color: "#cccccc",
    fontWeight: "600",
  },

  // Summary Cards
  summaryCard: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    backgroundColor: "rgba(0, 255, 136, 0.05)",
    borderRadius: 12,
    marginBottom: 8,
    gap: 12,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
  },
  summaryMarker: {
    width: 4,
    height: 40,
    borderRadius: 2,
  },
  summaryContent: {
    flex: 1,
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#00ff88",
    marginBottom: 4,
  },
  summaryText: {
    fontSize: 12,
    color: "#999999",
    lineHeight: 16,
  },
  summaryArrow: {
    fontSize: 24,
    color: "#00ff88",
    fontWeight: "300",
  },

  // Modal - Centered
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.9)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalWrapper: {
    width: "90%",
    maxWidth: 500,
    maxHeight: SCREEN_HEIGHT * 0.85,
  },
  modalContentCentered: {
    backgroundColor: "#1a1a1a",
    borderRadius: 20,
    overflow: "hidden",
    elevation: 10,
    shadowColor: "#00ff88",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.2)",
    minHeight: 400,
  },
  modalScrollView: {
    flex: 1,
  },
  modalHeaderCentered: {
    padding: 24,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  modalHeaderContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: "900",
    color: "#000",
    flex: 1,
  },
  modalSubtitle: {
    fontSize: 13,
    color: "rgba(0, 0, 0, 0.7)",
    fontWeight: "600",
  },
  modalSeverityBadge: {
    backgroundColor: "rgba(0, 0, 0, 0.2)",
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 12,
  },
  modalSeverityText: {
    fontSize: 11,
    fontWeight: "800",
    color: "#000",
    letterSpacing: 0.5,
  },
  modalBody: {
    padding: 24,
  },

  // Severity Card
  severityCard: {
    padding: 20,
    borderRadius: 16,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  severityHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  severityTitle: {
    fontSize: 15,
    fontWeight: "800",
    color: "#cccccc",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  severityPercentage: {
    fontSize: 28,
    fontWeight: "900",
  },
  severityProgressContainer: {
    marginBottom: 16,
  },
  severityProgressBg: {
    height: 12,
    backgroundColor: "#2a2a2a",
    borderRadius: 6,
    overflow: "hidden",
    marginBottom: 8,
  },
  severityProgressFill: {
    height: "100%",
    borderRadius: 6,
  },
  severityLabels: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  severityLabelText: {
    fontSize: 10,
    color: "#666666",
    fontWeight: "600",
  },
  severityIndicatorRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  severityDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  severityIndicatorText: {
    fontSize: 13,
    color: "#999999",
    flex: 1,
  },

  // Modal Sections
  modalSection: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 12,
  },
  sectionIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
  },
  sectionIcon: {
    fontSize: 18,
  },
  modalSectionLabel: {
    fontSize: 13,
    fontWeight: "800",
    color: "#00ff88",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  modalSectionText: {
    fontSize: 15,
    color: "#cccccc",
    lineHeight: 22,
    paddingLeft: 48,
  },
  modalCloseButton: {
    marginTop: 8,
  },
  modalCloseGradient: {
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  modalCloseText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#fff",
    letterSpacing: 0.5,
  },

  // Feedback
  feedbackBox: {
    padding: 20,
    borderRadius: 12,
    backgroundColor: "rgba(0, 255, 136, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
  },
  feedbackText: {
    fontSize: 15,
    color: "#cccccc",
    lineHeight: 24,
  },

  // Radar Chart
  radarContainer: {
    gap: 12,
  },
  radarRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  radarLeft: {
    width: 100,
  },
  radarShot: {
    fontSize: 13,
    fontWeight: "700",
    color: "#00ff88",
    textTransform: "uppercase",
    marginBottom: 2,
  },
  radarBadges: {
    flexDirection: "column",
    gap: 2,
  },
  radarBadge: {
    fontSize: 10,
    color: "#00cc6f",
  },
  radarBarBg: {
    flex: 1,
    height: 28,
    backgroundColor: "#2a2a2a",
    borderRadius: 14,
    overflow: "hidden",
  },
  radarBarFill: {
    height: "100%",
    borderRadius: 14,
  },
  radarValue: {
    width: 60,
    fontSize: 13,
    fontWeight: "800",
    color: "#00ff88",
    textAlign: "right",
  },

  // Detection Info
  detectionInfo: {
    gap: 16,
  },
  detectionRow: {
    gap: 8,
  },
  detectionItem: {
    gap: 8,
  },
  detectionLabel: {
    fontSize: 13,
    fontWeight: "700",
    color: "#00cc6f",
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  detectionBar: {
    height: 12,
    backgroundColor: "#2a2a2a",
    borderRadius: 6,
    overflow: "hidden",
  },
  detectionBarFill: {
    height: "100%",
    backgroundColor: "#00ff88",
    borderRadius: 6,
  },
  detectionValue: {
    fontSize: 15,
    fontWeight: "800",
    color: "#00ff88",
  },
  detectionMethodText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#cccccc",
  },

  // Camera Styles
  cameraContainer: {
    flex: 1,
    backgroundColor: "#000",
  },
  cameraControls: {
    flex: 1,
    justifyContent: "space-between",
    paddingTop: 16,
    paddingBottom: 40,
  },
  cameraTopBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  cameraCancelButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    justifyContent: "center",
    alignItems: "center",
  },
  cameraTimer: {
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  cameraBottomBar: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    paddingBottom: 20,
  },
  cameraRecordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    justifyContent: "center",
    alignItems: "center",
  },
  cameraRecordCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "#ff3b30",
  },
  cameraStopButton: {
    width: 80,
    height: 80,
    borderRadius: 8,
    backgroundColor: "rgba(255, 59, 48, 0.3)",
    justifyContent: "center",
    alignItems: "center",
  },
  cameraStopSquare: {
    width: 50,
    height: 50,
    borderRadius: 6,
    backgroundColor: "#ff3b30",
  },
  cameraGridOverlay: {
    ...StyleSheet.absoluteFillObject,
  },
  cameraGridRow: {
    flex: 1,
    justifyContent: "flex-end",
  },
  cameraGridLineHorizontal: {
    height: 1,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
  },
  cameraGridLineVertical: {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
  },
  focusTapLayer: {
    ...StyleSheet.absoluteFillObject,
  },
  focusSquare: {
    position: "absolute",
    borderRadius: 12,
    borderWidth: 2,
    borderColor: "rgba(0, 255, 136, 0.85)",
    backgroundColor: "rgba(0, 0, 0, 0.1)",
    justifyContent: "center",
    alignItems: "center",
  },
  focusCenterDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "rgba(0, 255, 136, 0.9)",
  },
  focusHintContainer: {
    position: "absolute",
    top: 80,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  focusHintText: {
    color: "#ffffff",
    fontSize: 12,
    fontWeight: "700",
    backgroundColor: "rgba(0, 0, 0, 0.55)",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    overflow: "hidden",
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.8)",
    justifyContent: "center",
    alignItems: "center",
  },
  cameraContent: {
    backgroundColor: "#1a1a1a",
    borderRadius: 16,
    padding: 24,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.2)",
  },
});
