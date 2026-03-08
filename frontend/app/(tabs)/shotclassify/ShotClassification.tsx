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
  Pressable,
  TouchableOpacity,
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
  quickCompareStances,
  analyzeStanceConsistency,
  StanceQuickCompareResult,
} from "../../../services/shotClassificationApi";
import { theme } from "../../../theme/theme";

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
  const size = SCREEN_WIDTH - 64;
  const center = size / 2;
  const radius = size / 2 - 46;
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
          stroke={theme.colors.neonAccent}
          strokeWidth="2"
        />

        <Polygon
          points={userPolygon}
          fill="rgba(255, 255, 255, 0.1)"
          stroke="rgba(255, 255, 255, 0.75)"
          strokeWidth="2"
        />

        {categories.map((cat, i) => {
          const labelPoint = getPoint(i, 106);
          return (
            <SvgText
              key={cat}
              x={labelPoint.x}
              y={labelPoint.y}
              fill={theme.colors.neonAccent}
              fontSize="10"
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
            style={[
              styles.spiderLegendDot,
              { backgroundColor: theme.colors.neonAccent },
            ]}
          />
          <Text style={styles.spiderLegendText}>Perfect Form</Text>
        </View>
        <View style={styles.spiderLegendItem}>
          <View
            style={[
              styles.spiderLegendDot,
              { backgroundColor: "rgba(255, 255, 255, 0.75)" },
            ]}
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
    const common = {
      color: theme.colors.neonAccent,
      bgColor: "rgba(0, 255, 136, 0.08)",
    };

    switch (severity.toLowerCase()) {
      case "critical":
        return {
          ...common,
          label: "CRITICAL",
          description: "Immediate attention required",
        };
      case "major":
        return {
          ...common,
          label: "MAJOR",
          description: "Significant improvement needed",
        };
      case "minor":
        return {
          ...common,
          label: "MINOR",
          description: "Fine-tuning recommended",
        };
      default:
        return {
          ...common,
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
              <View style={styles.modalHeaderCentered}>
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
                        { backgroundColor: "rgba(0, 255, 136, 0.1)" },
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
                    colors={[theme.colors.neonAccent, theme.colors.neonAccent]}
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
  const [stanceQuickResult, setStanceQuickResult] =
    useState<StanceQuickCompareResult | null>(null);
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
      const res = await quickCompareStances(stanceVideos[0], stanceVideos[1]);
      setStanceQuickResult(res);
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
      const res = await analyzeStanceConsistency(stanceVideos);
      setStanceAnalysisResult(res);
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

        {/* <View style={styles.heroStats}>
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
        </View> */}
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
                ? [theme.colors.neonAccent, theme.colors.neonAccent]
                : [theme.colors.navCard, theme.colors.navCard]
            }
            style={{ padding: 14, borderRadius: 40, alignItems: "center" }}
          >
            <Text
              style={{
                color:
                  activeTab === "shot"
                    ? theme.colors.background
                    : theme.colors.navInactiveText,
                fontWeight: "900",
                fontSize: 14,
                letterSpacing: 0.3,
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
                ? [theme.colors.neonAccent, theme.colors.neonAccent]
                : [theme.colors.navCard, theme.colors.navCard]
            }
            style={{ padding: 14, borderRadius: 40, alignItems: "center" }}
          >
            <Text
              style={{
                color:
                  activeTab === "stance"
                    ? theme.colors.background
                    : theme.colors.navInactiveText,
                fontWeight: "900",
                fontSize: 14,
                letterSpacing: 0.3,
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

              <View style={styles.uploadBlock}>
                <View style={styles.uploadBlockHeader}>
                  <Text style={styles.uploadBlockTitle}>STANCE VIDEOS</Text>
                  <View style={styles.uploadCountPill}>
                    <Text style={styles.uploadCountText}>
                      {stanceVideos.length}
                    </Text>
                  </View>
                </View>

                {stanceVideos.length === 0 ? (
                  <View style={styles.uploadEmptyState}>
                    <Text style={styles.uploadEmptyTitle}>
                      No videos added yet
                    </Text>
                    <Text style={styles.uploadEmptySubtext}>
                      Add at least 2 videos for consistency analysis
                    </Text>
                  </View>
                ) : (
                  <View style={styles.uploadVideoList}>
                    {stanceVideos.map((uri, idx) => (
                      <View key={idx} style={styles.uploadVideoRow}>
                        <View style={styles.uploadVideoIndexWrap}>
                          <Text style={styles.uploadVideoIndex}>{idx + 1}</Text>
                        </View>
                        <View style={styles.uploadVideoMeta}>
                          <Text style={styles.uploadVideoTitle}>
                            Video {idx + 1}
                          </Text>
                          <Text
                            style={styles.uploadVideoFilename}
                            numberOfLines={1}
                          >
                            {uri.split("/").pop()}
                          </Text>
                        </View>
                        <TouchableOpacity
                          onPress={() => removeStanceVideo(idx)}
                          activeOpacity={0.7}
                          style={styles.uploadRemoveButton}
                        >
                          <Text style={styles.uploadRemoveText}>Remove</Text>
                        </TouchableOpacity>
                      </View>
                    ))}
                  </View>
                )}

                <View style={styles.uploadActionsRow}>
                  <TouchableOpacity
                    onPress={pickStanceVideo}
                    activeOpacity={0.8}
                    style={styles.uploadActionSecondary}
                  >
                    <Text style={styles.uploadActionSecondaryText}>
                      Pick Video
                    </Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    onPress={handleTakeStanceVideo}
                    activeOpacity={0.8}
                    style={styles.uploadActionPrimaryWrap}
                  >
                    <LinearGradient
                      colors={[
                        theme.colors.neonAccent,
                        theme.colors.neonAccent,
                      ]}
                      style={styles.uploadActionPrimary}
                    >
                      <Text style={styles.uploadActionPrimaryText}>
                        Take Video
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
                    colors={[theme.colors.neonAccent, theme.colors.neonAccent]}
                    style={{
                      padding: 14,
                      borderRadius: 40,
                      alignItems: "center",
                      flexDirection: "row",
                      justifyContent: "center",
                      gap: 8,
                    }}
                  >
                    {stanceLoading ? (
                      <>
                        <ActivityIndicator
                          color={theme.colors.background}
                          size="small"
                        />
                        <Text
                          style={{
                            fontWeight: "800",
                            color: theme.colors.background,
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
                          color: theme.colors.background,
                          fontSize: 14,
                        }}
                      >
                        Analyze Consistency ({stanceVideos.length})
                      </Text>
                    )}
                  </LinearGradient>
                </TouchableOpacity>

                {stanceVideos.length < 2 && (
                  <Text
                    style={{
                      color: theme.colors.subtext,
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
                      color: theme.colors.neonAccent,
                      fontWeight: "900",
                      fontSize: 18,
                    }}
                  >
                    ← Back
                  </Text>
                </TouchableOpacity>
                <Text
                  style={{
                    marginLeft: 12,
                    color: "#FFFFFF",
                    fontWeight: "900",
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
                        borderColor: theme.colors.neonAccent,
                        backgroundColor: "rgba(0, 255, 136, 0.08)",
                        justifyContent: "center",
                        alignItems: "center",
                      }}
                    >
                      <Text
                        style={{
                          fontSize: 48,
                          fontWeight: "900",
                          color: theme.colors.neonAccent,
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
                          color: theme.colors.neonAccent,
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
                      <Text
                        style={{
                          color: theme.colors.neonAccent,
                          fontWeight: "800",
                        }}
                      >
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
                      <Text
                        style={{
                          color: theme.colors.neonAccent,
                          fontWeight: "800",
                        }}
                      >
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
                      <Text
                        style={{
                          color: theme.colors.neonAccent,
                          fontWeight: "800",
                        }}
                      >
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
                              color: theme.colors.neonAccent,
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
                            colors={[
                              theme.colors.neonAccent,
                              theme.colors.neonAccent,
                            ]}
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
                        borderLeftColor: theme.colors.neonAccent,
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
                        <Text style={{ color: theme.colors.neonAccent, fontWeight: "800" }}>{detail.stance_timing}</Text>
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
                              color: theme.colors.neonAccent,
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
                            colors={[
                              theme.colors.neonAccent,
                              theme.colors.neonAccent,
                            ]}
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
                      borderRadius: 16,
                      borderLeftWidth: 4,
                      borderLeftColor: theme.colors.neonAccent,
                    }}
                  >
                    <Text
                      style={{
                        color: theme.colors.neonAccent,
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
                      borderRadius: 16,
                      borderLeftWidth: 4,
                      borderLeftColor: theme.colors.neonAccent,
                    }}
                  >
                    <Text
                      style={{
                        color: theme.colors.neonAccent,
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
                      backgroundColor: "rgba(0, 255, 136, 0.05)",
                      borderRadius: 16,
                      borderLeftWidth: 4,
                      borderLeftColor: theme.colors.neonAccent,
                    }}
                  >
                    <Text
                      style={{
                        color: theme.colors.neonAccent,
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
                      backgroundColor: "rgba(0, 255, 136, 0.08)",
                      borderRadius: 16,
                      borderLeftWidth: 4,
                      borderLeftColor: theme.colors.neonAccent,
                    }}
                  >
                    <Text
                      style={{
                        color: theme.colors.neonAccent,
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
                          borderLeftColor: theme.colors.neonAccent,
                        }}
                      >
                        <Text
                          style={{
                            color: theme.colors.neonAccent,
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
                    colors={[theme.colors.neonAccent, theme.colors.neonAccent]}
                    style={{
                      padding: 14,
                      borderRadius: 40,
                      alignItems: "center",
                    }}
                  >
                    <Text
                      style={{
                        fontWeight: "800",
                        color: theme.colors.background,
                        fontSize: 14,
                      }}
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
                    style={[styles.shotChip, isActive && styles.shotChipActive]}
                  >
                    <Text
                      style={[
                        styles.shotChipText,
                        isSmallScreen && { fontSize: 11 },
                        isActive && styles.shotChipTextActive,
                      ]}
                      numberOfLines={2}
                    >
                      {shot.label}
                    </Text>
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

            <View style={styles.uploadBlock}>
              <View style={styles.uploadBlockHeader}>
                <Text style={styles.uploadBlockTitle}>SHOT VIDEO</Text>
                <View style={styles.uploadCountPill}>
                  <Text style={styles.uploadCountText}>{videoUri ? 1 : 0}</Text>
                </View>
              </View>

              {videoUri ? (
                <View style={styles.uploadVideoRow}>
                  <View style={styles.uploadVideoIndexWrap}>
                    <Text style={styles.uploadVideoIndex}>1</Text>
                  </View>
                  <View style={styles.uploadVideoMeta}>
                    <Text style={styles.uploadVideoTitle}>Video Selected</Text>
                    <Text style={styles.uploadVideoFilename} numberOfLines={1}>
                      {videoUri.split("/").pop()}
                    </Text>
                  </View>
                  <TouchableOpacity
                    onPress={() => setVideoUri(null)}
                    activeOpacity={0.7}
                    style={styles.uploadRemoveButton}
                  >
                    <Text style={styles.uploadRemoveText}>Remove</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                <View style={styles.uploadEmptyState}>
                  <Text style={styles.uploadEmptyTitle}>No video selected</Text>
                  <Text style={styles.uploadEmptySubtext}>
                    Pick from library or record a new shot video
                  </Text>
                </View>
              )}

              <View style={styles.uploadActionsRow}>
                <TouchableOpacity
                  onPress={pickVideo}
                  activeOpacity={0.8}
                  style={styles.uploadActionSecondary}
                >
                  <Text style={styles.uploadActionSecondaryText}>
                    Pick from Library
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={handleTakeShotVideo}
                  activeOpacity={0.8}
                  style={styles.uploadActionPrimaryWrap}
                >
                  <LinearGradient
                    colors={[theme.colors.neonAccent, theme.colors.neonAccent]}
                    style={styles.uploadActionPrimary}
                  >
                    <Text style={styles.uploadActionPrimaryText}>
                      Take Video
                    </Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
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
                colors={[theme.colors.neonAccent, theme.colors.neonAccent]}
                style={[
                  styles.analyzeButton,
                  (!videoUri || !selectedShot || analyzing) &&
                    styles.analyzeButtonDisabled,
                ]}
              >
                {analyzing ? (
                  <>
                    <ActivityIndicator
                      color={theme.colors.background}
                      size="small"
                    />
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
                      ? [theme.colors.neonAccent, theme.colors.neonAccent]
                      : [theme.colors.navCard, theme.colors.navCard]
                  }
                  style={styles.statusGradient}
                >
                  <Text
                    style={[
                      styles.statusText,
                      isSmallScreen && { fontSize: 14 },
                      {
                        color: result.is_correct
                          ? theme.colors.background
                          : "#FFFFFF",
                      },
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
                    <Text style={styles.scoreLabel}>INTENT SCORE</Text>
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
                          {
                            color: result.is_correct
                              ? theme.colors.neonAccent
                              : "rgba(255, 255, 255, 0.7)",
                          },
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
                                  ? theme.colors.neonAccent
                                  : mistake.severity === "major"
                                    ? theme.colors.neonAccent
                                    : mistake.severity === "minor"
                                      ? theme.colors.neonAccent
                                      : theme.colors.neonAccent,
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
                              ? [
                                  theme.colors.neonAccent,
                                  theme.colors.neonAccent,
                                ]
                              : item.shot === result.intended_shot
                                ? [
                                    theme.colors.neonAccent,
                                    theme.colors.neonAccent,
                                  ]
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
    backgroundColor: theme.colors.background,
  },

  // Hero
  hero: {
    paddingBottom: theme.spacing.xl,
    paddingHorizontal: theme.spacing.m,
    backgroundColor: theme.colors.background,
    paddingTop: theme.spacing.xl,
  },
  heroContent: {
    alignItems: "center",
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: "900",
    color: "#FFFFFF",
    letterSpacing: -0.5,
    textAlign: "center",
    marginBottom: theme.spacing.s,
  },
  heroSubtitle: {
    fontSize: 13,
    fontWeight: "600",
    color: theme.colors.subtext,
    textAlign: "center",
    textTransform: "uppercase",
    letterSpacing: 2,
  },
  heroStats: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginTop: theme.spacing.l,
    paddingHorizontal: theme.spacing.m,
  },
  stat: {
    alignItems: "center",
  },
  statValue: {
    fontSize: 24,
    fontWeight: "900",
    color: "rgba(255, 255, 255, 0.95)",
    letterSpacing: -0.5,
  },
  statLabel: {
    fontSize: 10,
    color: theme.colors.subtext,
    textTransform: "uppercase",
    letterSpacing: 2,
    marginTop: theme.spacing.s,
    fontWeight: "800",
  },

  // Card
  card: {
    backgroundColor: theme.colors.navCard,
    borderRadius: 40,
    padding: theme.spacing.l,
    marginHorizontal: theme.spacing.m,
    marginBottom: theme.spacing.l,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.05)",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: theme.spacing.l,
    paddingHorizontal: theme.spacing.s,
  },
  cardHeaderText: {
    flex: 1,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#FFFFFF",
    marginBottom: theme.spacing.s,
    letterSpacing: -0.5,
  },
  cardSubtitle: {
    fontSize: 13,
    color: theme.colors.subtext,
    fontWeight: "500",
  },
  chartHint: {
    fontSize: 11,
    color: "rgba(255, 255, 255, 0.6)",
    textAlign: "center",
    marginBottom: theme.spacing.m,
    fontWeight: "500",
  },

  // Shot Selection
  shotGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    rowGap: theme.spacing.s,
    columnGap: theme.spacing.s,
    paddingHorizontal: 0,
  },
  shotChip: {
    width: "48.5%",
    minHeight: 50,
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.14)",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255, 255, 255, 0.02)",
  },
  shotChipActive: {
    borderColor: theme.colors.neonAccent,
    backgroundColor: "rgba(0, 255, 136, 0.12)",
  },
  shotChipText: {
    fontSize: 13,
    fontWeight: "700",
    color: "rgba(255, 255, 255, 0.88)",
    letterSpacing: 0,
    textAlign: "center",
    lineHeight: 18,
  },
  shotChipTextActive: {
    color: "#FFFFFF",
    fontWeight: "800",
  },
  checkMarkContainer: {
    position: "absolute",
    top: -6,
    right: -6,
    backgroundColor: theme.colors.background,
    borderRadius: 10,
    width: 20,
    height: 20,
    alignItems: "center",
    justifyContent: "center",
    ...Platform.select({
      ios: {
        shadowColor: theme.colors.neonAccent,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.4,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  checkMark: {
    color: theme.colors.neonAccent,
    fontSize: 14,
    fontWeight: "bold",
  },

  // Upload
  uploadArea: {
    padding: theme.spacing.l,
    borderRadius: 20,
    alignItems: "center",
  },
  uploadContent: {
    alignItems: "center",
    gap: theme.spacing.m,
  },
  uploadTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: theme.colors.background,
    marginBottom: theme.spacing.s,
    letterSpacing: -0.5,
  },
  uploadSubtitle: {
    fontSize: 13,
    color: `${theme.colors.background}B3`,
    fontWeight: "500",
  },
  uploadBlock: {
    gap: theme.spacing.m,
  },
  uploadBlockHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  uploadBlockTitle: {
    fontSize: 12,
    fontWeight: "800",
    color: theme.colors.neonAccent,
    letterSpacing: 1.2,
  },
  uploadCountPill: {
    minWidth: 28,
    height: 28,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0, 255, 136, 0.16)",
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.45)",
    paddingHorizontal: 8,
  },
  uploadCountText: {
    fontSize: 12,
    fontWeight: "900",
    color: theme.colors.neonAccent,
  },
  uploadEmptyState: {
    borderWidth: 1,
    borderStyle: "dashed",
    borderColor: "rgba(255, 255, 255, 0.2)",
    borderRadius: 16,
    paddingVertical: 20,
    paddingHorizontal: 14,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, 0.02)",
    gap: 4,
  },
  uploadEmptyTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "rgba(255, 255, 255, 0.86)",
  },
  uploadEmptySubtext: {
    fontSize: 12,
    fontWeight: "500",
    color: "rgba(255, 255, 255, 0.56)",
    textAlign: "center",
  },
  uploadVideoList: {
    gap: theme.spacing.s,
  },
  uploadVideoRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(0, 255, 136, 0.06)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.16)",
    paddingVertical: 10,
    paddingHorizontal: 10,
    gap: 10,
  },
  uploadVideoIndexWrap: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(0, 255, 136, 0.2)",
  },
  uploadVideoIndex: {
    fontSize: 12,
    fontWeight: "900",
    color: theme.colors.neonAccent,
  },
  uploadVideoMeta: {
    flex: 1,
    gap: 2,
  },
  uploadVideoTitle: {
    fontSize: 13,
    fontWeight: "800",
    color: theme.colors.neonAccent,
  },
  uploadVideoFilename: {
    fontSize: 11,
    fontWeight: "500",
    color: "rgba(255, 255, 255, 0.62)",
  },
  uploadRemoveButton: {
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.18)",
    backgroundColor: "rgba(255, 255, 255, 0.04)",
    borderRadius: 999,
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  uploadRemoveText: {
    fontSize: 11,
    fontWeight: "700",
    color: "rgba(255, 255, 255, 0.78)",
  },
  uploadActionsRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing.s,
  },
  uploadActionPrimaryWrap: {
    flex: 1,
  },
  uploadActionPrimary: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 13,
    borderRadius: 14,
  },
  uploadActionPrimaryText: {
    fontSize: 13,
    fontWeight: "900",
    color: theme.colors.background,
    letterSpacing: 0.2,
  },
  uploadActionSecondary: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 13,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
    backgroundColor: "rgba(255, 255, 255, 0.03)",
  },
  uploadActionSecondaryText: {
    fontSize: 13,
    fontWeight: "800",
    color: "rgba(255, 255, 255, 0.9)",
  },

  // Analyze Button
  analyzeButtonContainer: {
    marginHorizontal: theme.spacing.m,
    marginBottom: theme.spacing.l,
    marginTop: theme.spacing.m,
  },
  analyzeButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 20,
    borderRadius: 40,
    gap: theme.spacing.m,
    ...Platform.select({
      ios: {
        shadowColor: theme.colors.neonAccent,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
      },
      android: {
        elevation: 8,
      },
    }),
  },
  analyzeButtonDisabled: {
    opacity: 0.5,
  },
  analyzeButtonText: {
    fontSize: 18,
    fontWeight: "900",
    color: theme.colors.background,
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },

  // Status Badge
  statusBadge: {
    marginHorizontal: theme.spacing.m,
    marginBottom: theme.spacing.l,
  },
  statusGradient: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.l,
    borderRadius: 40,
  },
  statusText: {
    fontSize: 16,
    fontWeight: "800",
    color: theme.colors.text,
    letterSpacing: 0.5,
    textTransform: "uppercase",
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
    borderColor: theme.colors.neonAccent,
    backgroundColor: "rgba(0, 255, 136, 0.08)",
    justifyContent: "center",
    alignItems: "center",
  },
  scoreValue: {
    fontSize: 40,
    fontWeight: "900",
    color: theme.colors.neonAccent,
    letterSpacing: -1,
  },
  scoreLabel: {
    fontSize: 9,
    color: theme.colors.neonAccent,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    fontWeight: "800",
    textAlign: "center",
    lineHeight: 11,
    includeFontPadding: false,
  },
  scoreInfo: {
    flex: 1,
    gap: theme.spacing.s,
  },
  scoreRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 14,
    backgroundColor: "rgba(0, 255, 136, 0.08)",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
  },
  scoreRowLabel: {
    fontSize: 13,
    color: theme.colors.subtext,
    fontWeight: "600",
  },
  scoreRowValue: {
    fontSize: 14,
    fontWeight: "800",
    color: theme.colors.neonAccent,
    textTransform: "uppercase",
    letterSpacing: -0.3,
  },

  // Spider Chart
  spiderContainer: {
    alignItems: "center",
    paddingVertical: theme.spacing.l,
    paddingHorizontal: theme.spacing.m,
  },
  spiderLegend: {
    flexDirection: "row",
    justifyContent: "center",
    gap: theme.spacing.l,
    marginTop: theme.spacing.l,
  },
  spiderLegendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing.s,
  },
  spiderLegendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  spiderLegendText: {
    fontSize: 12,
    color: "rgba(255, 255, 255, 0.8)",
    fontWeight: "600",
  },

  // Summary Cards
  summaryCard: {
    flexDirection: "row",
    alignItems: "center",
    padding: theme.spacing.l,
    backgroundColor: "rgba(0, 255, 136, 0.08)",
    borderRadius: 16,
    marginBottom: theme.spacing.m,
    marginHorizontal: theme.spacing.m,
    gap: theme.spacing.m,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
  },
  summaryMarker: {
    width: 4,
    height: 40,
    borderRadius: 2,
    backgroundColor: theme.colors.neonAccent,
  },
  summaryContent: {
    flex: 1,
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#FFFFFF",
    marginBottom: theme.spacing.s,
    letterSpacing: -0.3,
  },
  summaryText: {
    fontSize: 12,
    color: "rgba(255, 255, 255, 0.65)",
    lineHeight: 18,
    fontWeight: "500",
  },
  summaryArrow: {
    fontSize: 24,
    color: theme.colors.neonAccent,
    fontWeight: "300",
  },

  // Modal - Centered
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.92)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalWrapper: {
    width: "88%",
    maxWidth: 500,
    maxHeight: SCREEN_HEIGHT * 0.85,
  },
  modalContentCentered: {
    backgroundColor: theme.colors.navCard,
    borderRadius: 28,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.05)",
    minHeight: 400,
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 12 },
        shadowOpacity: 0.5,
        shadowRadius: 32,
      },
      android: {
        elevation: 15,
      },
    }),
  },
  modalScrollView: {
    flex: 1,
  },
  modalHeaderCentered: {
    padding: theme.spacing.l,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    backgroundColor: theme.colors.navCard,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255, 255, 255, 0.05)",
  },
  modalHeaderContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: theme.spacing.s,
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: "900",
    color: "rgba(255, 255, 255, 0.95)",
    flex: 1,
    letterSpacing: -0.5,
  },
  modalSubtitle: {
    fontSize: 13,
    color: "rgba(255, 255, 255, 0.7)",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  modalSeverityBadge: {
    backgroundColor: "rgba(0, 255, 136, 0.1)",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.2)",
  },
  modalSeverityText: {
    fontSize: 11,
    fontWeight: "800",
    color: theme.colors.neonAccent,
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  modalBody: {
    padding: theme.spacing.l,
    gap: theme.spacing.l,
  },

  // Severity Card
  severityCard: {
    padding: theme.spacing.l,
    borderRadius: 20,
    marginBottom: theme.spacing.l,
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
    backgroundColor: "rgba(0, 255, 136, 0.05)",
  },
  severityHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: theme.spacing.l,
  },
  severityTitle: {
    fontSize: 15,
    fontWeight: "800",
    color: "rgba(255, 255, 255, 0.85)",
    textTransform: "uppercase",
    letterSpacing: 2,
  },
  severityPercentage: {
    fontSize: 32,
    fontWeight: "900",
    color: theme.colors.neonAccent,
    letterSpacing: -1,
  },
  severityProgressContainer: {
    marginBottom: theme.spacing.l,
  },
  severityProgressBg: {
    height: 10,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    borderRadius: 5,
    overflow: "hidden",
    marginBottom: theme.spacing.s,
  },
  severityProgressFill: {
    height: "100%",
    borderRadius: 5,
  },
  severityLabels: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  severityLabelText: {
    fontSize: 10,
    color: theme.colors.subtext,
    fontWeight: "700",
  },
  severityIndicatorRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing.m,
    paddingVertical: theme.spacing.s,
  },
  severityDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  severityIndicatorText: {
    fontSize: 13,
    color: "rgba(255, 255, 255, 0.7)",
    flex: 1,
    fontWeight: "500",
  },

  // Modal Sections
  modalSection: {
    marginBottom: theme.spacing.l,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing.m,
    marginBottom: theme.spacing.m,
  },
  sectionIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0, 255, 136, 0.1)",
  },
  sectionIcon: {
    fontSize: 20,
    color: theme.colors.neonAccent,
  },
  modalSectionLabel: {
    fontSize: 13,
    fontWeight: "800",
    color: "#FFFFFF",
    textTransform: "uppercase",
    letterSpacing: 2,
  },
  modalSectionText: {
    fontSize: 15,
    color: "rgba(255, 255, 255, 0.8)",
    lineHeight: 24,
    paddingLeft: 48,
    fontWeight: "500",
  },
  modalCloseButton: {
    marginTop: theme.spacing.l,
  },
  modalCloseGradient: {
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: theme.colors.neonAccent,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  modalCloseText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#fff",
    letterSpacing: 1,
    textTransform: "uppercase",
  },

  // Feedback
  feedbackBox: {
    padding: theme.spacing.l,
    borderRadius: 18,
    backgroundColor: "rgba(0, 255, 136, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
    marginHorizontal: theme.spacing.m,
  },
  feedbackText: {
    fontSize: 15,
    color: "rgba(255, 255, 255, 0.85)",
    lineHeight: 24,
    fontWeight: "500",
  },

  // Radar Chart
  radarContainer: {
    gap: theme.spacing.l,
    paddingHorizontal: theme.spacing.m,
  },
  radarRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: theme.spacing.m,
  },
  radarLeft: {
    width: 100,
  },
  radarShot: {
    fontSize: 13,
    fontWeight: "700",
    color: "#FFFFFF",
    textTransform: "uppercase",
    marginBottom: theme.spacing.s,
    letterSpacing: -0.3,
  },
  radarBadges: {
    flexDirection: "column",
    gap: 4,
  },
  radarBadge: {
    fontSize: 10,
    color: theme.colors.neonAccent,
    fontWeight: "600",
  },
  radarBarBg: {
    flex: 1,
    height: 28,
    backgroundColor: "rgba(0, 0, 0, 0.2)",
    borderRadius: 14,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(0, 255, 136, 0.1)",
  },
  radarBarFill: {
    height: "100%",
    borderRadius: 14,
  },
  radarValue: {
    width: 60,
    fontSize: 13,
    fontWeight: "800",
    color: theme.colors.neonAccent,
    textAlign: "right",
    letterSpacing: -0.3,
  },

  // Detection Info
  detectionInfo: {
    gap: theme.spacing.l,
    paddingHorizontal: theme.spacing.m,
  },
  detectionRow: {
    gap: theme.spacing.s,
  },
  detectionItem: {
    gap: theme.spacing.s,
  },
  detectionLabel: {
    fontSize: 13,
    fontWeight: "700",
    color: "#FFFFFF",
    textTransform: "uppercase",
    letterSpacing: 2,
  },
  detectionBar: {
    height: 12,
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    borderRadius: 6,
    overflow: "hidden",
    borderWidth: 0.5,
    borderColor: "rgba(0, 255, 136, 0.2)",
  },
  detectionBarFill: {
    height: "100%",
    backgroundColor: theme.colors.neonAccent,
    borderRadius: 6,
  },
  detectionValue: {
    fontSize: 15,
    fontWeight: "800",
    color: theme.colors.neonAccent,
    letterSpacing: -0.3,
  },
  detectionMethodText: {
    fontSize: 14,
    fontWeight: "600",
    color: "rgba(255, 255, 255, 0.8)",
  },

  // Camera Styles
  cameraContainer: {
    flex: 1,
    backgroundColor: "#000",
  },
  cameraControls: {
    flex: 1,
    justifyContent: "space-between",
    paddingTop: theme.spacing.m,
    paddingBottom: theme.spacing.xl,
  },
  cameraTopBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: theme.spacing.l,
    paddingVertical: theme.spacing.m,
  },
  cameraCancelButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    justifyContent: "center",
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.5,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  cameraTimer: {
    backgroundColor: "rgba(0, 0, 0, 0.7)",
    paddingHorizontal: theme.spacing.m,
    paddingVertical: theme.spacing.s,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  cameraBottomBar: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    paddingBottom: theme.spacing.l,
  },
  cameraRecordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    justifyContent: "center",
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 8,
      },
      android: {
        elevation: 4,
      },
    }),
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
    borderRadius: 12,
    backgroundColor: "rgba(255, 59, 48, 0.2)",
    justifyContent: "center",
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 8,
      },
      android: {
        elevation: 4,
      },
    }),
  },
  cameraStopSquare: {
    width: 50,
    height: 50,
    borderRadius: 8,
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
    backgroundColor: "rgba(255, 255, 255, 0.15)",
  },
  cameraGridLineVertical: {
    position: "absolute",
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
  },
  focusTapLayer: {
    ...StyleSheet.absoluteFillObject,
  },
  focusSquare: {
    position: "absolute",
    borderRadius: 14,
    borderWidth: 2.5,
    borderColor: `${theme.colors.neonAccent}E6`,
    backgroundColor: `${theme.colors.neonAccent}0D`,
    justifyContent: "center",
    alignItems: "center",
    ...Platform.select({
      ios: {
        shadowColor: theme.colors.neonAccent,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.3,
        shadowRadius: 6,
      },
    }),
  },
  focusCenterDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.neonAccent,
  },
  focusHintContainer: {
    position: "absolute",
    top: 120,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  focusHintText: {
    color: "#ffffff",
    fontSize: 12,
    fontWeight: "700",
    backgroundColor: "rgba(0, 0, 0, 0.65)",
    paddingHorizontal: theme.spacing.m,
    paddingVertical: theme.spacing.s,
    borderRadius: 10,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  cameraOverlay: {
    flex: 1,
    backgroundColor: "rgba(0, 0, 0, 0.85)",
    justifyContent: "center",
    alignItems: "center",
  },
  cameraContent: {
    backgroundColor: theme.colors.navCard,
    borderRadius: 24,
    padding: theme.spacing.l,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.05)",
    ...Platform.select({
      ios: {
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
      },
      android: {
        elevation: 8,
      },
    }),
  },
});
