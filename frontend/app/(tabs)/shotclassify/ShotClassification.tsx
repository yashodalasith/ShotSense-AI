/**
 * Shot Classification Screen - React Native
 * Mobile-responsive with 2D Vector Cricket Player
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
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import Svg, {
  Circle,
  Rect,
  Path,
  Line,
  Polygon,
  G,
  Defs,
  LinearGradient as SvgGradient,
  Stop,
  TSpan,
  Text as SvgText,
} from "react-native-svg";
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

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");
const isSmallScreen = SCREEN_HEIGHT < 700;

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

interface Avatar2DProps {
  keypoints: Keypoint3D[];
  mistakes: Mistake[];
  highlightedJoint: string | null;
}

// Convert 3D keypoints to 2D screen coordinates
const projectTo2D = (keypoint: Keypoint3D, width: number, height: number) => {
  // Normalize 3D coordinates to 2D with some perspective
  const x = (keypoint.position.x + 50) * (width / 100);
  const y = (keypoint.position.y + 50) * (height / 100);
  return { x, y };
};

// Calculate angle between two points
const calculateAngle = (
  p1: { x: number; y: number },
  p2: { x: number; y: number }
) => {
  return Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
};

// Get joint position from keypoints with fallback
const getJointPosition = (
  keypoints: Keypoint3D[],
  jointName: string,
  fallback: { x: number; y: number }
) => {
  const joint = keypoints.find((kp) => kp.joint === jointName);
  if (joint) {
    return {
      x: joint.position.x,
      y: joint.position.y,
      z: joint.position.z,
    };
  }
  return { x: fallback.x, y: fallback.y, z: 0 };
};

// 2D Vector Cricket Player Component (Fixed)
const VectorCricketPlayer: React.FC<Avatar2DProps> = ({
  keypoints,
  mistakes,
  highlightedJoint,
}) => {
  const containerWidth = SCREEN_WIDTH - 64;
  const containerHeight = isSmallScreen ? 280 : 340;

  // Center of the SVG
  const centerX = containerWidth / 2;
  const centerY = containerHeight / 2;

  // Simple cricket player with basic shapes
  const playerHeight = 180;
  const playerWidth = 80;

  return (
    <View style={[styles.vectorPlayerContainer, { height: containerHeight }]}>
      <Svg width={containerWidth} height={containerHeight}>
        {/* Green ground */}
        <Rect
          x="0"
          y={containerHeight - 60}
          width={containerWidth}
          height="60"
          fill="#90EE90"
        />

        {/* Brown pitch */}
        <Rect
          x={centerX - 80}
          y={containerHeight - 60}
          width="160"
          height="60"
          fill="#D2B48C"
        />

        {/* Stumps */}
        <Rect
          x={centerX + 100}
          y={containerHeight - 110}
          width="10"
          height="50"
          fill="#8B4513"
        />
        <Rect
          x={centerX + 115}
          y={containerHeight - 110}
          width="10"
          height="50"
          fill="#8B4513"
        />
        <Rect
          x={centerX + 130}
          y={containerHeight - 110}
          width="10"
          height="50"
          fill="#8B4513"
        />

        {/* Bails */}
        <Rect
          x={centerX + 105}
          y={containerHeight - 110}
          width="30"
          height="5"
          fill="#FFFFFF"
        />

        {/* Player Body */}

        {/* Left Leg (Front) */}
        <Rect
          x={centerX - 25}
          y={centerY + 40}
          width="15"
          height="80"
          fill="#00008B"
          rx="5"
        />

        {/* Left Pad */}
        <Rect
          x={centerX - 30}
          y={centerY + 50}
          width="25"
          height="60"
          fill="#2F4F4F"
          rx="3"
        />

        {/* Right Leg (Back) */}
        <Rect
          x={centerX + 10}
          y={centerY + 20}
          width="15"
          height="100"
          fill="#00008B"
          rx="5"
        />

        {/* Right Pad */}
        <Rect
          x={centerX + 5}
          y={centerY + 30}
          width="25"
          height="80"
          fill="#2F4F4F"
          rx="3"
        />

        {/* Body Torso */}
        <Rect
          x={centerX - 25}
          y={centerY - 60}
          width="50"
          height="100"
          fill="#00008B"
          rx="10"
        />

        {/* Left Arm (Front) */}
        <Rect
          x={centerX - 40}
          y={centerY - 50}
          width="15"
          height="50"
          fill="#00008B"
          rx="5"
        />

        {/* Left Glove */}
        <Circle
          cx={centerX - 32}
          cy={centerY}
          r="12"
          fill="#FFFFFF"
          stroke="#000"
          strokeWidth="2"
        />

        {/* Right Arm (Back) */}
        <Rect
          x={centerX + 25}
          y={centerY - 70}
          width="15"
          height="70"
          fill="#00008B"
          rx="5"
        />

        {/* Right Glove */}
        <Circle
          cx={centerX + 32}
          cy={centerY}
          r="12"
          fill="#FFFFFF"
          stroke="#000"
          strokeWidth="2"
        />

        {/* Head */}
        <Circle cx={centerX} cy={centerY - 90} r="20" fill="#FFCC99" />

        {/* Face */}
        <Circle cx={centerX - 6} cy={centerY - 95} r="2" fill="#000" />
        <Circle cx={centerX + 6} cy={centerY - 95} r="2" fill="#000" />
        <Path
          d={`M ${centerX - 8} ${centerY - 80} Q ${centerX} ${centerY - 75}, ${
            centerX + 8
          } ${centerY - 80}`}
          stroke="#000"
          strokeWidth="2"
          fill="none"
        />

        {/* Helmet */}
        <Circle
          cx={centerX}
          cy={centerY - 90}
          r="22"
          fill="none"
          stroke="#00008B"
          strokeWidth="8"
          strokeDasharray="20,10"
        />

        {/* Bat */}
        <Rect
          x={centerX + 32}
          y={centerY - 150}
          width="8"
          height="120"
          fill="#8B4513"
          rx="3"
        />

        {/* Bat Blade */}
        <Rect
          x={centerX + 40}
          y={centerY - 180}
          width="4"
          height="80"
          fill="#D2691E"
        />
        <Rect
          x={centerX + 38}
          y={centerY - 180}
          width="8"
          height="20"
          fill="#D2691E"
        />

        {/* Shoes */}
        <Rect
          x={centerX - 30}
          y={centerY + 115}
          width="20"
          height="10"
          fill="#FFFFFF"
          rx="3"
          stroke="#000"
          strokeWidth="1"
        />
        <Rect
          x={centerX + 10}
          y={centerY + 115}
          width="20"
          height="10"
          fill="#FFFFFF"
          rx="3"
          stroke="#000"
          strokeWidth="1"
        />

        {/* Mistake Indicators */}
        {mistakes.map((mistake, index) => {
          // Map joint_id to positions
          const jointPositions: { [key: string]: { x: number; y: number }[] } =
            {
              shoulder_rotation: [
                { x: centerX - 25, y: centerY - 50 },
                { x: centerX + 25, y: centerY - 50 },
              ],
              front_elbow: [{ x: centerX + 32, y: centerY - 20 }],
              back_elbow: [{ x: centerX - 32, y: centerY - 20 }],
              front_wrist: [{ x: centerX + 32, y: centerY + 20 }],
              back_wrist: [{ x: centerX - 32, y: centerY + 20 }],
              front_knee: [{ x: centerX + 18, y: centerY + 70 }],
              back_knee: [{ x: centerX - 18, y: centerY + 90 }],
              torso_bend: [{ x: centerX, y: centerY - 10 }],
              body_position: [{ x: centerX, y: centerY - 10 }],
              Torso: [{ x: centerX, y: centerY - 10 }],
            };

          const affectedJoints = jointPositions[mistake.joint_id] || [
            { x: centerX, y: centerY },
          ];

          return affectedJoints.map((joint, jointIndex) => {
            const isHighlighted = highlightedJoint === mistake.joint_id;
            const severityColor = mistake.severity_color;

            return (
              <G key={`${index}-${jointIndex}`}>
                {/* Glow effect when highlighted */}
                {isHighlighted && (
                  <Circle
                    cx={joint.x}
                    cy={joint.y}
                    r="25"
                    fill={severityColor}
                    opacity="0.3"
                  />
                )}

                {/* Joint indicator */}
                <Circle
                  cx={joint.x}
                  cy={joint.y}
                  r="12"
                  fill={severityColor}
                  stroke="#FFFFFF"
                  strokeWidth="3"
                />

                {/* Severity text */}
                <SvgText
                  x={joint.x}
                  y={joint.y + 4}
                  fill="#FFFFFF"
                  fontSize="10"
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {mistake.severity === "critical"
                    ? "C"
                    : mistake.severity === "major"
                    ? "M"
                    : mistake.severity === "minor"
                    ? "m"
                    : "n"}
                </SvgText>

                {/* Arrow pointer */}
                <Path
                  d={`M ${joint.x + 30} ${joint.y - 30} L ${joint.x + 5} ${
                    joint.y - 5
                  }`}
                  stroke={severityColor}
                  strokeWidth="3"
                  fill="none"
                />
                <Circle
                  cx={joint.x + 30}
                  cy={joint.y - 30}
                  r="5"
                  fill={severityColor}
                />
              </G>
            );
          });
        })}
      </Svg>
    </View>
  );
};

export default function ShotClassificationScreen() {
  const [shotTypes, setShotTypes] = useState<ShotType[]>([]);
  const [selectedShot, setSelectedShot] = useState<string | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [score, setScore] = useState(0);
  const [highlightedJoint, setHighlightedJoint] = useState<string | null>(null);

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

  // Calculate radar chart data
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
      {/* Hero Section with Cricket Icons */}
      <LinearGradient
        colors={["rgba(99, 102, 241, 0.12)", "rgba(139, 92, 246, 0.08)"]}
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
            { value: "2D", label: "Visual", icon: "👁️" },
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
                      ? ["#6366f1", "#8b5cf6"]
                      : ["rgba(99, 102, 241, 0.06)", "rgba(139, 92, 246, 0.04)"]
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
                ? ["#10b981", "#34d399"]
                : ["rgba(99, 102, 241, 0.06)", "rgba(139, 92, 246, 0.06)"]
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
                  ? ["#10b981", "#34d399"]
                  : ["#f59e0b", "#fbbf24"]
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
                      {
                        color: result.is_correct ? "#10b981" : "#f59e0b",
                      },
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
                    ⚡ Contact Frame:
                  </Text>
                  <Text
                    style={[
                      styles.scoreRowValue,
                      isSmallScreen && { fontSize: 12 },
                    ]}
                  >
                    #{result.analysis_metadata.contact_frame}
                  </Text>
                </View>
              </View>
            </View>
          </Animated.View>

          {/* 2D Vector Cricket Player */}
          <Animated.View entering={FadeInUp.delay(200)} style={styles.card}>
            <View style={styles.cardHeader}>
              <Text
                style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
              >
                👤
              </Text>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
                >
                  Batting Pose Analysis
                </Text>
                <Text
                  style={[
                    styles.cardSubtitle,
                    isSmallScreen && { fontSize: 12 },
                  ]}
                >
                  Visual representation of your shot mechanics
                </Text>
              </View>
            </View>

            <VectorCricketPlayer
              keypoints={result.visual_feedback.keypoints_3d.actual}
              mistakes={result.visual_feedback.mistakes}
              highlightedJoint={highlightedJoint}
            />

            <Text style={styles.playerInstructions}>
              💡 Tap on mistakes below to highlight the affected joints
            </Text>
          </Animated.View>

          {/* Hotspot Mistake Cards */}
          {result.visual_feedback.mistakes.length > 0 && (
            <Animated.View entering={FadeInUp.delay(300)} style={styles.card}>
              <View style={styles.cardHeader}>
                <Text
                  style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
                >
                  ⚠️
                </Text>
                <View style={styles.cardHeaderText}>
                  <Text
                    style={[
                      styles.cardTitle,
                      isSmallScreen && { fontSize: 16 },
                    ]}
                  >
                    Technical Issues Found
                  </Text>
                  <Text
                    style={[
                      styles.cardSubtitle,
                      isSmallScreen && { fontSize: 12 },
                    ]}
                  >
                    {result.visual_feedback.mistakes.length} area
                    {result.visual_feedback.mistakes.length > 1 ? "s" : ""} need
                    improvement
                  </Text>
                </View>
              </View>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.mistakesScroll}
              >
                {result.visual_feedback.mistakes.map((mistake, idx) => (
                  <TouchableOpacity
                    key={idx}
                    onPress={() =>
                      setHighlightedJoint(
                        highlightedJoint === mistake.joint_id
                          ? null
                          : mistake.joint_id
                      )
                    }
                    activeOpacity={0.7}
                  >
                    <LinearGradient
                      colors={
                        highlightedJoint === mistake.joint_id
                          ? [mistake.severity_color, mistake.severity_color]
                          : ["#ffffff", "#f8fafc"]
                      }
                      style={[
                        styles.mistakeCard,
                        {
                          width: isSmallScreen
                            ? SCREEN_WIDTH * 0.85
                            : SCREEN_WIDTH * 0.75,
                        },
                        {
                          borderColor: mistake.severity_color,
                          borderWidth:
                            highlightedJoint === mistake.joint_id ? 3 : 2,
                        },
                      ]}
                    >
                      <View
                        style={[
                          styles.mistakeCardHeader,
                          {
                            backgroundColor:
                              highlightedJoint === mistake.joint_id
                                ? "rgba(255,255,255,0.3)"
                                : mistake.severity_color,
                          },
                        ]}
                      >
                        <View style={styles.mistakeTitleContainer}>
                          <Text
                            style={[
                              styles.mistakeCardTitle,
                              isSmallScreen && { fontSize: 14 },
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "#ffffff"
                                    : mistake.severity === "minor"
                                    ? "#1e293b"
                                    : "#ffffff",
                              },
                            ]}
                          >
                            {mistake.body_part.toUpperCase()}
                          </Text>
                          <Text
                            style={[
                              styles.mistakeCardSubtitle,
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "rgba(255,255,255,0.8)"
                                    : mistake.severity === "minor"
                                    ? "#475569"
                                    : "rgba(255,255,255,0.9)",
                              },
                            ]}
                          >
                            {mistake.joint_id.replace(/_/g, " ")}
                          </Text>
                        </View>
                        <View
                          style={[
                            styles.mistakeSeverityBadge,
                            {
                              backgroundColor:
                                highlightedJoint === mistake.joint_id
                                  ? "rgba(255,255,255,0.4)"
                                  : "rgba(0,0,0,0.1)",
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.mistakeSeverityBadgeText,
                              isSmallScreen && { fontSize: 9 },
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "#ffffff"
                                    : mistake.severity === "minor"
                                    ? "#1e293b"
                                    : "#ffffff",
                              },
                            ]}
                          >
                            {mistake.severity.toUpperCase()}
                          </Text>
                        </View>
                      </View>

                      <View style={styles.mistakeCardBody}>
                        <View style={styles.mistakeSection}>
                          <Text
                            style={[
                              styles.mistakeCardLabel,
                              isSmallScreen && { fontSize: 10 },
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "#ffffff"
                                    : "#64748b",
                              },
                            ]}
                          >
                            ❌ ISSUE:
                          </Text>
                          <Text
                            style={[
                              styles.mistakeCardText,
                              isSmallScreen && { fontSize: 12, lineHeight: 18 },
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "#ffffff"
                                    : "#1e293b",
                              },
                            ]}
                          >
                            {mistake.explanation}
                          </Text>
                        </View>

                        <View style={styles.mistakeSection}>
                          <Text
                            style={[
                              styles.mistakeCardLabel,
                              isSmallScreen && { fontSize: 10 },
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "#ffffff"
                                    : "#64748b",
                              },
                            ]}
                          >
                            ✅ FIX:
                          </Text>
                          <Text
                            style={[
                              styles.mistakeCardText,
                              isSmallScreen && { fontSize: 12, lineHeight: 18 },
                              {
                                color:
                                  highlightedJoint === mistake.joint_id
                                    ? "#ffffff"
                                    : "#1e293b",
                              },
                            ]}
                          >
                            {mistake.recommendation}
                          </Text>
                        </View>
                      </View>

                      <View style={styles.glowIndicator}>
                        <Text
                          style={[
                            styles.glowLabel,
                            isSmallScreen && { fontSize: 10 },
                            {
                              color:
                                highlightedJoint === mistake.joint_id
                                  ? "#ffffff"
                                  : "#64748b",
                            },
                          ]}
                        >
                          ISSUE SEVERITY:
                        </Text>
                        <View
                          style={[
                            styles.glowBar,
                            {
                              backgroundColor:
                                highlightedJoint === mistake.joint_id
                                  ? "rgba(255,255,255,0.3)"
                                  : "#e2e8f0",
                            },
                          ]}
                        >
                          <View
                            style={[
                              styles.glowBarFill,
                              {
                                width: `${mistake.glow_intensity * 100}%`,
                                backgroundColor: mistake.severity_color,
                              },
                            ]}
                          />
                        </View>
                        <View style={styles.glowScale}>
                          <Text
                            style={[
                              styles.glowScaleText,
                              highlightedJoint === mistake.joint_id && {
                                color: "#ffffff",
                              },
                            ]}
                          >
                            Low
                          </Text>
                          <Text
                            style={[
                              styles.glowScaleText,
                              highlightedJoint === mistake.joint_id && {
                                color: "#ffffff",
                              },
                            ]}
                          >
                            Medium
                          </Text>
                          <Text
                            style={[
                              styles.glowScaleText,
                              highlightedJoint === mistake.joint_id && {
                                color: "#ffffff",
                              },
                            ]}
                          >
                            High
                          </Text>
                        </View>
                      </View>
                    </LinearGradient>
                  </TouchableOpacity>
                ))}
              </ScrollView>
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
              colors={["rgba(99, 102, 241, 0.08)", "rgba(139, 92, 246, 0.06)"]}
              style={styles.feedbackBox}
            >
              <Text
                style={[styles.feedbackText, isSmallScreen && { fontSize: 13 }]}
              >
                {result.coaching_feedback}
              </Text>
            </LinearGradient>
          </Animated.View>

          {/* Radar Chart - Shot Confusion */}
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
                          ? ["#6366f1", "#8b5cf6"]
                          : item.shot === result.intended_shot
                          ? ["#10b981", "#34d399"]
                          : ["#94a3b8", "#cbd5e1"]
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

          {/* Timeline */}
          <Animated.View entering={FadeInUp.delay(600)} style={styles.card}>
            <View style={styles.cardHeader}>
              <Text
                style={[styles.cardIcon, isSmallScreen && { fontSize: 24 }]}
              >
                ⏱️
              </Text>
              <View style={styles.cardHeaderText}>
                <Text
                  style={[styles.cardTitle, isSmallScreen && { fontSize: 16 }]}
                >
                  Shot Contact Timeline
                </Text>
                <Text
                  style={[
                    styles.cardSubtitle,
                    isSmallScreen && { fontSize: 12 },
                  ]}
                >
                  Ball impact detected at frame #
                  {result.analysis_metadata.contact_frame}
                </Text>
              </View>
            </View>

            <View style={styles.timeline}>
              <View style={styles.timelineBar}>
                <View
                  style={[
                    styles.timelineMarker,
                    {
                      left: `${
                        (result.analysis_metadata.contact_frame / 30) * 100
                      }%`,
                    },
                  ]}
                >
                  <LinearGradient
                    colors={["#f59e0b", "#fbbf24"]}
                    style={styles.timelineMarkerCircle}
                  >
                    <Text style={styles.timelineMarkerText}>💥</Text>
                  </LinearGradient>
                  <Text
                    style={[
                      styles.timelineMarkerLabel,
                      isSmallScreen && { fontSize: 10 },
                    ]}
                  >
                    BALL CONTACT
                  </Text>
                </View>
              </View>

              <View style={styles.timelineLabels}>
                <Text
                  style={[
                    styles.timelineLabel,
                    isSmallScreen && { fontSize: 11 },
                  ]}
                >
                  START
                </Text>
                <Text
                  style={[
                    styles.timelineLabel,
                    isSmallScreen && { fontSize: 11 },
                  ]}
                >
                  FRAME #{result.analysis_metadata.contact_frame}
                </Text>
                <Text
                  style={[
                    styles.timelineLabel,
                    isSmallScreen && { fontSize: 11 },
                  ]}
                >
                  END
                </Text>
              </View>
            </View>
          </Animated.View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8fafc",
  },

  // Hero
  hero: {
    paddingBottom: 40,
    paddingHorizontal: 20,
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
    color: "#6366f1",
    letterSpacing: 1.5,
    textAlign: "center",
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 14,
    color: "#8b5cf6",
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
    shadowColor: "#6366f1",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 4,
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
    color: "#1e293b",
    marginBottom: 4,
  },
  cardSubtitle: {
    fontSize: 13,
    color: "#8b5cf6",
  },

  // Vector Player Container
  vectorPlayerContainer: {
    width: "100%",
    borderRadius: 16,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "rgba(99, 102, 241, 0.1)",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  playerInstructions: {
    fontSize: 12,
    color: "#64748b",
    textAlign: "center",
    marginTop: 12,
    fontStyle: "italic",
  },
  jointLabels: {
    position: "absolute",
    pointerEvents: "none",
  },
  jointLabel: {
    position: "absolute",
    backgroundColor: "rgba(99, 102, 241, 0.8)",
    padding: 4,
    borderRadius: 4,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 40,
    minHeight: 40,
  },
  jointLabelText: {
    color: "#ffffff",
    fontSize: 8,
    textAlign: "center",
    fontWeight: "bold",
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
    elevation: 6,
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
    color: "#1e293b",
    marginBottom: 4,
  },
  uploadSubtitle: {
    fontSize: 13,
    color: "#0a0a0aff",
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
  },
  analyzeButtonText: {
    fontSize: 16,
    fontWeight: "800",
    color: "#fff",
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
    color: "#fff",
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
    borderColor: "#e0e7ff",
    backgroundColor: "rgba(99, 102, 241, 0.05)",
    justifyContent: "center",
    alignItems: "center",
  },
  scoreValue: {
    fontSize: 36,
    fontWeight: "900",
    color: "#6366f1",
  },
  scoreLabel: {
    fontSize: 10,
    color: "#8b5cf6",
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
    backgroundColor: "rgba(99, 102, 241, 0.05)",
    borderRadius: 8,
  },
  scoreRowLabel: {
    fontSize: 13,
    color: "#64748b",
    fontWeight: "600",
  },
  scoreRowValue: {
    fontSize: 14,
    fontWeight: "800",
    color: "#1e293b",
    textTransform: "uppercase",
  },

  // Mistake Cards
  mistakesScroll: {
    paddingRight: 16,
  },
  mistakeCard: {
    borderRadius: 16,
    marginRight: 12,
    overflow: "hidden",
  },
  mistakeCardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 16,
  },
  mistakeTitleContainer: {
    flex: 1,
  },
  mistakeCardTitle: {
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 2,
  },
  mistakeCardSubtitle: {
    fontSize: 10,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  mistakeSeverityBadge: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 12,
    marginLeft: 8,
  },
  mistakeSeverityBadgeText: {
    fontSize: 10,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  mistakeCardBody: {
    padding: 16,
    paddingTop: 8,
  },
  mistakeSection: {
    marginBottom: 12,
  },
  mistakeCardLabel: {
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 6,
  },
  mistakeCardText: {
    fontSize: 14,
    lineHeight: 20,
  },
  glowIndicator: {
    padding: 16,
    paddingTop: 0,
  },
  glowLabel: {
    fontSize: 11,
    fontWeight: "600",
    marginBottom: 8,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  glowBar: {
    height: 8,
    borderRadius: 4,
    overflow: "hidden",
    marginBottom: 4,
  },
  glowBarFill: {
    height: "100%",
    borderRadius: 4,
  },
  glowScale: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  glowScaleText: {
    fontSize: 10,
    color: "#64748b",
  },

  // Feedback
  feedbackBox: {
    padding: 20,
    borderRadius: 12,
  },
  feedbackText: {
    fontSize: 15,
    color: "#1e293b",
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
    color: "#1e293b",
    textTransform: "uppercase",
    marginBottom: 2,
  },
  radarBadges: {
    flexDirection: "column",
    gap: 2,
  },
  radarBadge: {
    fontSize: 10,
    color: "#8b5cf6",
  },
  radarBarBg: {
    flex: 1,
    height: 28,
    backgroundColor: "#f1f5f9",
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
    color: "#1e293b",
    textAlign: "right",
  },

  // Timeline
  timeline: {
    paddingVertical: 20,
  },
  timelineBar: {
    height: 8,
    backgroundColor: "#e2e8f0",
    borderRadius: 4,
    position: "relative",
  },
  timelineMarker: {
    position: "absolute",
    top: -12,
    alignItems: "center",
  },
  timelineMarkerCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#f59e0b",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  timelineMarkerText: {
    fontSize: 16,
  },
  timelineMarkerLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: "#f59e0b",
    marginTop: 8,
  },
  timelineLabels: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 32,
  },
  timelineLabel: {
    fontSize: 12,
    color: "#64748b",
    fontWeight: "600",
  },
});
