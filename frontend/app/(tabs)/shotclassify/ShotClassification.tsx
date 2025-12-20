/**
 * Shot Classification Screen
 * Intent-Based Shot Scoring Component
 */

import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import Animated, {
  FadeInUp,
  useSharedValue,
  useAnimatedStyle,
  withTiming,
} from "react-native-reanimated";

import {
  getShotTypes,
  analyzeShot,
  ShotType,
  AnalysisResult,
} from "../../../services/shotClassificationApi";

export default function ShotClassificationScreen() {
  const [shotTypes, setShotTypes] = useState<ShotType[]>([]);
  const [selectedShot, setSelectedShot] = useState<string | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const scoreAnim = useSharedValue(0);

  useEffect(() => {
    loadShotTypes();
    ImagePicker.requestMediaLibraryPermissionsAsync();
  }, []);

  useEffect(() => {
    if (result) {
      scoreAnim.value = withTiming(result.intent_score, { duration: 800 });
    }
  }, [result]);

  const loadShotTypes = async () => {
    try {
      setLoading(true);
      setShotTypes(await getShotTypes());
    } catch {
      Alert.alert("Error", "Failed to load shot types");
    } finally {
      setLoading(false);
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
    }
  };

  const handleAnalyze = async () => {
    if (!videoUri || !selectedShot) return;

    try {
      setAnalyzing(true);
      setResult(await analyzeShot(videoUri, selectedShot));
    } catch {
      Alert.alert("Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  const scoreStyle = useAnimatedStyle(() => ({
    transform: [{ scale: withTiming(result ? 1 : 0.8) }],
  }));

  return (
    <ScrollView style={styles.container}>
      {/* HERO */}
      <LinearGradient colors={["#fff", "#ffe4ec"]} style={styles.hero}>
        <Text style={styles.heroTitle}>SHOT INTELLIGENCE</Text>
        <Text style={styles.heroSub}>AI-powered cricket analysis</Text>
      </LinearGradient>

      {/* SHOT SELECT */}
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>INTENDED SHOT</Text>

        {loading ? (
          <ActivityIndicator color="#ff4081" />
        ) : (
          <View style={styles.shotGrid}>
            {shotTypes.map((shot) => {
              const active = selectedShot === shot.value;
              return (
                <TouchableOpacity
                  key={shot.value}
                  onPress={() => setSelectedShot(shot.value)}
                  activeOpacity={0.8}
                >
                  <LinearGradient
                    colors={
                      active ? ["#ff1744", "#ff4081"] : ["#ffffff", "#ffffff"]
                    }
                    style={[styles.shotChip, active && styles.glow]}
                  >
                    <Text
                      style={[styles.shotText, active && styles.shotTextActive]}
                    >
                      {shot.label}
                    </Text>
                  </LinearGradient>
                </TouchableOpacity>
              );
            })}
          </View>
        )}
      </View>

      {/* UPLOAD */}
      <TouchableOpacity style={styles.uploadCard} onPress={pickVideo}>
        <LinearGradient
          colors={["#ff1744", "#ff4081"]}
          style={styles.uploadGradient}
        >
          <Text style={styles.uploadTitle}>
            {videoUri ? "VIDEO READY" : "UPLOAD SHOT VIDEO"}
          </Text>
          <Text style={styles.uploadSub}>Slow-motion preferred</Text>
        </LinearGradient>
      </TouchableOpacity>

      {/* ANALYZE */}
      <TouchableOpacity
        disabled={!videoUri || !selectedShot || analyzing}
        onPress={handleAnalyze}
        style={{ marginTop: 20 }}
      >
        <LinearGradient
          colors={["#d50000", "#ff1744"]}
          style={styles.analyzeButton}
        >
          {analyzing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.analyzeText}>RUN AI ANALYSIS</Text>
          )}
        </LinearGradient>
      </TouchableOpacity>

      {/* RESULT */}
      {result && (
        <Animated.View entering={FadeInUp.duration(600)} style={styles.card}>
          <Text style={styles.sectionTitle}>AI VERDICT</Text>

          <Animated.Text style={[styles.score, scoreStyle]}>
            {result.intent_score}%
          </Animated.Text>
          <Text style={styles.scoreLabel}>Intent Accuracy</Text>

          <Text style={styles.feedback}>{result.feedback}</Text>

          {Object.entries(result.probability_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([shot, prob]) => (
              <Animated.View
                key={shot}
                entering={FadeInUp.delay(100)}
                style={styles.barRow}
              >
                <Text style={styles.barLabel}>{shot}</Text>
                <View style={styles.barBg}>
                  <Animated.View
                    style={[styles.barFill, { width: `${prob * 100}%` }]}
                  />
                </View>
                <Text style={styles.barVal}>{(prob * 100).toFixed(0)}%</Text>
              </Animated.View>
            ))}
        </Animated.View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fff" },

  hero: {
    paddingTop: 80,
    paddingBottom: 50,
    alignItems: "center",
  },
  heroTitle: {
    fontSize: 30,
    fontWeight: "900",
    color: "#b71c1c",
    letterSpacing: 2,
  },
  heroSub: { color: "#ad1457", marginTop: 6 },

  card: {
    backgroundColor: "#fff",
    margin: 20,
    borderRadius: 22,
    padding: 20,
    shadowColor: "#ff4081",
    shadowOpacity: 0.2,
    shadowRadius: 24,
    elevation: 8,
  },

  sectionTitle: {
    color: "#b71c1c",
    fontWeight: "800",
    letterSpacing: 1,
    marginBottom: 14,
  },

  shotGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  shotChip: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: "#ffd6e1",
  },
  shotText: { color: "#b71c1c", fontWeight: "700" },
  shotTextActive: { color: "#fff" },
  glow: {
    shadowColor: "#ff4081",
    shadowOpacity: 0.9,
    shadowRadius: 18,
  },

  uploadCard: { marginHorizontal: 20, borderRadius: 24, overflow: "hidden" },
  uploadGradient: { paddingVertical: 34, alignItems: "center" },
  uploadTitle: { color: "#fff", fontWeight: "900" },
  uploadSub: { color: "#ffe4ec", fontSize: 12 },

  analyzeButton: {
    marginHorizontal: 20,
    paddingVertical: 18,
    borderRadius: 30,
    alignItems: "center",
    shadowColor: "#ff1744",
    shadowOpacity: 0.4,
    shadowRadius: 20,
    elevation: 8,
  },
  analyzeText: {
    color: "#fff",
    fontWeight: "900",
    letterSpacing: 1,
  },

  score: {
    fontSize: 56,
    fontWeight: "900",
    color: "#d50000",
    textAlign: "center",
  },
  scoreLabel: {
    textAlign: "center",
    color: "#ad1457",
    marginBottom: 16,
  },

  feedback: { color: "#333", marginBottom: 14 },

  barRow: { flexDirection: "row", alignItems: "center", marginTop: 10 },
  barLabel: { width: 70, color: "#ad1457", fontSize: 12 },
  barBg: {
    flex: 1,
    height: 10,
    backgroundColor: "#ffe4ec",
    borderRadius: 6,
    marginHorizontal: 8,
  },
  barFill: {
    height: "100%",
    backgroundColor: "#ff4081",
    borderRadius: 6,
  },
  barVal: {
    width: 40,
    color: "#b71c1c",
    fontSize: 12,
    textAlign: "right",
  },
});
