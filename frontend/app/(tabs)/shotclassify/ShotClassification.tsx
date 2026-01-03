/**
 * Shot Classification Screen - React Native
 * Mobile-responsive with Spider Chart Visualization
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
  Dimensions,
  Modal,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import Animated, {
  FadeInUp,
  FadeInDown,
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

  // Map mistake analysis to categories
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
        m.body_part.toLowerCase().includes(cat.toLowerCase())
      )
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
    getPoint(i, getCategoryScore(cat))
  );

  const perfectPolygon = perfectPoints.map((p) => `${p.x},${p.y}`).join(" ");
  const userPolygon = userPoints.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <View style={styles.spiderContainer}>
      <Svg width={size} height={size}>
        {/* Background circles */}
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

        {/* Axes */}
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

        {/* Perfect form (outer) */}
        <Polygon
          points={perfectPolygon}
          fill="rgba(0, 255, 136, 0.1)"
          stroke="#00ff88"
          strokeWidth="2"
        />

        {/* User's execution (inner) */}
        <Polygon
          points={userPolygon}
          fill="rgba(255, 59, 48, 0.2)"
          stroke="#ff3b30"
          strokeWidth="2"
        />

        {/* Category labels */}
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

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.modalOverlay}>
        <TouchableOpacity
          style={styles.modalBackground}
          activeOpacity={1}
          onPress={onClose}
        />

        <Animated.View
          entering={FadeInUp.duration(300)}
          style={styles.modalContent}
        >
          <LinearGradient
            colors={["#00ff88", "#00cc6f"]}
            style={styles.modalHeader}
          >
            <View style={styles.modalHeaderContent}>
              <Text style={styles.modalTitle}>
                {mistake.body_part.toUpperCase()}
              </Text>
              <View style={styles.modalSeverityBadge}>
                <Text style={styles.modalSeverityText}>
                  {mistake.severity.toUpperCase()}
                </Text>
              </View>
            </View>
          </LinearGradient>

          <View style={styles.modalBody}>
            <View style={styles.modalSection}>
              <Text style={styles.modalSectionIcon}>❌</Text>
              <View style={styles.modalSectionContent}>
                <Text style={styles.modalSectionLabel}>THE ISSUE</Text>
                <Text style={styles.modalSectionText}>
                  {mistake.explanation}
                </Text>
              </View>
            </View>

            <View style={styles.modalSection}>
              <Text style={styles.modalSectionIcon}>✅</Text>
              <View style={styles.modalSectionContent}>
                <Text style={styles.modalSectionLabel}>HOW TO FIX IT</Text>
                <Text style={styles.modalSectionText}>
                  {mistake.recommendation}
                </Text>
              </View>
            </View>

            <View style={styles.modalSeverityBar}>
              <Text style={styles.modalSeverityBarLabel}>ISSUE SEVERITY</Text>
              <View style={styles.severityBarContainer}>
                <View
                  style={[
                    styles.severityBarFill,
                    {
                      width: `${mistake.glow_intensity * 100}%`,
                      backgroundColor: "#00ff88",
                    },
                  ]}
                />
              </View>
              <View style={styles.severityScale}>
                <Text style={styles.severityScaleText}>Low</Text>
                <Text style={styles.severityScaleText}>Medium</Text>
                <Text style={styles.severityScaleText}>High</Text>
              </View>
            </View>

            <TouchableOpacity
              onPress={onClose}
              activeOpacity={0.8}
              style={styles.modalCloseButton}
            >
              <LinearGradient
                colors={["#00ff88", "#00cc6f"]}
                style={styles.modalCloseGradient}
              >
                <Text style={styles.modalCloseText}>Got it!</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>
        </Animated.View>
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
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Hero Section */}
      <LinearGradient
        colors={["rgba(0, 255, 136, 0.15)", "rgba(0, 204, 111, 0.1)"]}
        style={[styles.hero, { paddingTop: isSmallScreen ? 40 : 60 }]}
      >
        <Animated.View
          entering={FadeInDown.duration(800)}
          style={styles.heroContent}
        >
          <View style={styles.heroIconRow}>
            <Text style={styles.heroIcon}>🏏</Text>
            <Text style={styles.heroIcon}>🎯</Text>
            <Text style={styles.heroIcon}>📊</Text>
          </View>
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
            { value: "99.2%", label: "Accuracy", icon: "🎯" },
            { value: "Visual", label: "Feedback", icon: "👁️" },
            { value: "AI", label: "Powered", icon: "🤖" },
          ].map((stat, idx) => (
            <Animated.View
              key={idx}
              entering={FadeInUp.delay(idx * 100).duration(600)}
              style={styles.stat}
            >
              <Text
                style={[styles.statIcon, isSmallScreen && { fontSize: 24 }]}
              >
                {stat.icon}
              </Text>
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

      {/* Shot Selection */}
      <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}>
            🏏
          </Text>
          <View style={styles.cardHeaderText}>
            <Text style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}>
              Select Intended Shot
            </Text>
            <Text
              style={[styles.cardSubtitle, isSmallScreen && { fontSize: 12 }]}
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
                      : ["rgba(0, 255, 136, 0.1)", "rgba(0, 204, 111, 0.05)"]
                  }
                  style={[styles.shotChip, isActive && styles.shotChipActive]}
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
                    <Text
                      style={[
                        styles.checkMark,
                        isSmallScreen && { fontSize: 14 },
                      ]}
                    >
                      ✓
                    </Text>
                  )}
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
                ? ["#00ff88", "#00cc6f"]
                : ["rgba(0, 255, 136, 0.1)", "rgba(0, 204, 111, 0.1)"]
            }
            style={styles.uploadArea}
          >
            <Text
              style={[
                styles.uploadIconLarge,
                isSmallScreen && { fontSize: 40 },
              ]}
            >
              {videoUri ? "✅" : "📹"}
            </Text>
            <View style={styles.uploadText}>
              <Text
                style={[styles.uploadTitle, isSmallScreen && { fontSize: 16 }]}
              >
                {videoUri ? "Video Selected!" : "Upload Batting Video"}
              </Text>
              <Text
                style={[
                  styles.uploadSubtitle,
                  isSmallScreen && { fontSize: 11 },
                ]}
              >
                {videoUri ? "Tap to change" : "Select your batting video"}
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
              <>
                <Text style={styles.analyzeIcon}>🚀</Text>
                <Text
                  style={[
                    styles.analyzeButtonText,
                    isSmallScreen && { fontSize: 14 },
                  ]}
                >
                  Analyze with AI
                </Text>
              </>
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
              <Text style={styles.statusIcon}>
                {result.is_correct ? "✅" : "⚠️"}
              </Text>
              <Text
                style={[styles.statusText, isSmallScreen && { fontSize: 14 }]}
              >
                {result.is_correct ? "PERFECT FORM!" : "FORM MISMATCH DETECTED"}
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
                  style={[styles.scoreValue, isSmallScreen && { fontSize: 30 }]}
                >
                  {score}%
                </Text>
                <Text
                  style={[styles.scoreLabel, isSmallScreen && { fontSize: 9 }]}
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
                    🎯 Intended:
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
                    🔍 Detected:
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
                    📊 Prototype:
                  </Text>
                  <Text
                    style={[
                      styles.scoreRowValue,
                      isSmallScreen && { fontSize: 12 },
                    ]}
                  >
                    {result.analysis_metadata.prototype_samples} samples
                  </Text>
                </View>
              </View>
            </View>
          </Animated.View>

          {/* Spider Chart - Shot DNA */}
          <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
            <View style={styles.cardHeader}>
              <Text
                style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
              >
                🕸️
              </Text>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
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
          </Animated.View>

          {/* Mistake Summary Cards */}
          {result?.visual_feedback.mistakes &&
            result.visual_feedback.mistakes.length > 0 && (
              <Animated.View entering={FadeInUp.delay(300)} style={styles.card}>
                <View style={styles.cardHeader}>
                  <Text
                    style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
                  >
                    📋
                  </Text>
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
                        { backgroundColor: "#00ff88" },
                      ]}
                    >
                      <Text style={styles.summaryMarkerText}>
                        {mistake.severity === "critical"
                          ? "!"
                          : mistake.severity === "major"
                          ? "⚠"
                          : "•"}
                      </Text>
                    </View>
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
              <Text
                style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
              >
                💬
              </Text>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
                >
                  AI Cricket Coach Feedback
                </Text>
                <Text
                  style={[
                    styles.cardSubtitle,
                    isSmallScreen && { fontSize: 12 },
                  ]}
                >
                  Personalized recommendations from our AI coach
                </Text>
              </View>
            </View>
            <LinearGradient
              colors={["rgba(0, 255, 136, 0.1)", "rgba(0, 204, 111, 0.08)"]}
              style={styles.feedbackBox}
            >
              <Text
                style={[styles.feedbackText, isSmallScreen && { fontSize: 13 }]}
              >
                {result.coaching_feedback}
              </Text>
            </LinearGradient>
          </Animated.View>

          {/* Radar Chart */}
          <Animated.View entering={FadeInUp.delay(500)} style={styles.card}>
            <View style={styles.cardHeader}>
              <Text
                style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
              >
                📊
              </Text>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
                >
                  Shot Classification Analysis
                </Text>
                <Text
                  style={[
                    styles.cardSubtitle,
                    isSmallScreen && { fontSize: 12 },
                  ]}
                >
                  AI confidence distribution across shot types
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
                    style={[styles.radarLeft, isSmallScreen && { width: 80 }]}
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
                          🎯 INTENDED
                        </Text>
                      )}
                      {item.shot === result.predicted_shot && (
                        <Text
                          style={[
                            styles.radarBadge,
                            isSmallScreen && { fontSize: 9 },
                          ]}
                        >
                          🔍 DETECTED
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
                      style={[styles.radarBarFill, { width: `${item.value}%` }]}
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
          <Animated.View entering={FadeInUp.delay(600)} style={styles.card}>
            <View style={styles.cardHeader}>
              <Text
                style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
              >
                🎥
              </Text>
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
          </Animated.View>
        </>
      )}

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
  heroIconRow: {
    flexDirection: "row",
    gap: 16,
    marginBottom: 16,
  },
  heroIcon: {
    fontSize: 32,
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
  statIcon: {
    fontSize: 28,
    marginBottom: 8,
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
    borderRadius: 20,
    padding: 20,
    marginHorizontal: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#2a2a2a",
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
    gap: 12,
  },
  cardIcon: {
    fontSize: 28,
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
    color: "#00cc6f",
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
  checkMark: {
    position: "absolute",
    top: 4,
    right: 4,
    color: "#000",
    fontSize: 16,
  },

  // Upload
  uploadArea: {
    flexDirection: "row",
    alignItems: "center",
    padding: 24,
    borderRadius: 16,
    gap: 16,
  },
  uploadIconLarge: {
    fontSize: 48,
  },
  uploadText: {
    flex: 1,
  },
  uploadTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#d0ff00e0",
    marginBottom: 4,
  },
  uploadSubtitle: {
    fontSize: 13,
    color: "#1a1a1aff",
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
    gap: 8,
  },
  analyzeButtonDisabled: {
    opacity: 0.5,
  },
  analyzeIcon: {
    fontSize: 20,
  },
  analyzeButtonText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#000",
    letterSpacing: 1,
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
    borderRadius: 16,
    gap: 8,
  },
  statusIcon: {
    fontSize: 20,
  },
  statusText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#000",
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
    borderWidth: 10,
    borderColor: "#2a2a2a",
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
    paddingVertical: 8,
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
    padding: 12,
    backgroundColor: "rgba(0, 255, 136, 0.05)",
    borderRadius: 12,
    marginBottom: 8,
    gap: 12,
  },
  summaryMarker: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
  },
  summaryMarkerText: {
    color: "#000",
    fontSize: 18,
    fontWeight: "bold",
  },
  summaryContent: {
    flex: 1,
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: "#00ff88",
    marginBottom: 2,
  },
  summaryText: {
    fontSize: 12,
    color: "#cccccc",
    lineHeight: 16,
  },
  summaryArrow: {
    fontSize: 24,
    color: "#666666",
  },

  // Modal
  modalOverlay: {
    flex: 1,
    justifyContent: "flex-end",
  },
  modalBackground: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.8)",
  },
  modalContent: {
    backgroundColor: "#1a1a1a",
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    maxHeight: "80%",
  },
  modalHeader: {
    padding: 20,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
  },
  modalHeaderContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: "900",
    color: "#000",
    flex: 1,
  },
  modalSeverityBadge: {
    backgroundColor: "rgba(0, 0, 0, 0.3)",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
  },
  modalSeverityText: {
    fontSize: 11,
    fontWeight: "700",
    color: "#000",
  },
  modalBody: {
    padding: 20,
  },
  modalSection: {
    flexDirection: "row",
    marginBottom: 20,
    gap: 12,
  },
  modalSectionIcon: {
    fontSize: 24,
  },
  modalSectionContent: {
    flex: 1,
  },
  modalSectionLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: "#00cc6f",
    marginBottom: 6,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  modalSectionText: {
    fontSize: 14,
    color: "#cccccc",
    lineHeight: 20,
  },
  modalSeverityBar: {
    marginBottom: 20,
  },
  modalSeverityBarLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: "#666666",
    marginBottom: 8,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  severityBarContainer: {
    height: 8,
    backgroundColor: "#2a2a2a",
    borderRadius: 4,
    overflow: "hidden",
    marginBottom: 6,
  },
  severityBarFill: {
    height: "100%",
  },
  severityScale: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  severityScaleText: {
    fontSize: 10,
    color: "#666666",
  },
  modalCloseButton: {
    marginTop: 8,
  },
  modalCloseGradient: {
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
  },
  modalCloseText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#000",
  },

  // Feedback
  feedbackBox: {
    padding: 20,
    borderRadius: 12,
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
    justifyContent: "center",
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
});
