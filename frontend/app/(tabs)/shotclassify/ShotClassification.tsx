/**
 * Shot Classification Screen
 * Intent-Based Shot Scoring Component
 */

import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
  Image,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import {
  getShotTypes,
  analyzeShot,
  ShotType,
  AnalysisResult,
} from "../../../services/shotClassificationApi";

export default function ShotClassificationScreen() {
  const [shotTypes, setShotTypes] = useState<ShotType[]>([]);
  const [selectedShot, setSelectedShot] = useState<string>("");
  const [videoUri, setVideoUri] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    loadShotTypes();
    requestPermissions();
  }, []);

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission Required",
        "Camera roll permission is required to select videos"
      );
    }
  };

  const loadShotTypes = async () => {
    try {
      setLoading(true);
      const types = await getShotTypes();
      setShotTypes(types);
    } catch (error) {
      Alert.alert(
        "Error",
        "Failed to load shot types. Please check your connection."
      );
    } finally {
      setLoading(false);
    }
  };

  const pickVideo = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 1,
      });

      if (!result.canceled && result.assets[0]) {
        setVideoUri(result.assets[0].uri);
        setResult(null); // Clear previous results
      }
    } catch (error) {
      Alert.alert("Error", "Failed to pick video");
    }
  };

  const handleAnalyze = async () => {
    if (!videoUri) {
      Alert.alert("No Video", "Please select a video first");
      return;
    }

    if (!selectedShot) {
      Alert.alert("No Shot Selected", "Please select your intended shot type");
      return;
    }

    try {
      setAnalyzing(true);
      const analysisResult = await analyzeShot(videoUri, selectedShot);
      setResult(analysisResult);
    } catch (error) {
      Alert.alert(
        "Analysis Failed",
        "Could not analyze the shot. Please try again."
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "#2ecc71";
    if (score >= 60) return "#f39c12";
    if (score >= 40) return "#e67e22";
    return "#e74c3c";
  };

  const getFeedbackEmoji = (score: number) => {
    if (score >= 80) return "🎯";
    if (score >= 60) return "👍";
    if (score >= 40) return "🤔";
    return "📉";
  };

  return (
    <ScrollView style={styles.container}>
      <LinearGradient
        colors={["#1a237e", "#283593", "#3949ab"]}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <Ionicons name="baseball" size={50} color="#fff" />
          <Text style={styles.headerTitle}>Shot Classification</Text>
          <Text style={styles.headerSubtitle}>
            Intent-Based Shot Scoring Analysis
          </Text>
        </View>
      </LinearGradient>

      <View style={styles.content}>
        {/* Step 1: Select Shot Type */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Step 1: Select Intended Shot</Text>
          <Text style={styles.sectionSubtitle}>
            Choose the shot you plan to play
          </Text>

          {loading ? (
            <ActivityIndicator size="large" color="#1a237e" />
          ) : (
            <View style={styles.shotTypeGrid}>
              {shotTypes.map((shot) => (
                <TouchableOpacity
                  key={shot.value}
                  style={[
                    styles.shotTypeButton,
                    selectedShot === shot.value &&
                      styles.shotTypeButtonSelected,
                  ]}
                  onPress={() => setSelectedShot(shot.value)}
                >
                  <Text
                    style={[
                      styles.shotTypeText,
                      selectedShot === shot.value &&
                        styles.shotTypeTextSelected,
                    ]}
                  >
                    {shot.label}
                  </Text>
                  {selectedShot === shot.value && (
                    <Ionicons name="checkmark-circle" size={20} color="#fff" />
                  )}
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>

        {/* Step 2: Upload Video */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Step 2: Upload Video</Text>
          <Text style={styles.sectionSubtitle}>
            Record or select a video of your shot
          </Text>

          <TouchableOpacity style={styles.uploadButton} onPress={pickVideo}>
            <LinearGradient
              colors={["#3949ab", "#1a237e"]}
              style={styles.uploadButtonGradient}
            >
              <Ionicons name="videocam" size={40} color="#fff" />
              <Text style={styles.uploadButtonText}>
                {videoUri ? "Change Video" : "Select Video"}
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          {videoUri && (
            <View style={styles.videoPreview}>
              <Ionicons name="checkmark-circle" size={30} color="#2ecc71" />
              <Text style={styles.videoSelectedText}>Video Selected</Text>
            </View>
          )}
        </View>

        {/* Step 3: Analyze Button */}
        <TouchableOpacity
          style={[
            styles.analyzeButton,
            (!videoUri || !selectedShot) && styles.analyzeButtonDisabled,
          ]}
          onPress={handleAnalyze}
          disabled={!videoUri || !selectedShot || analyzing}
        >
          <LinearGradient
            colors={
              !videoUri || !selectedShot
                ? ["#bdc3c7", "#95a5a6"]
                : ["#2ecc71", "#27ae60"]
            }
            style={styles.analyzeButtonGradient}
          >
            {analyzing ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <>
                <Ionicons name="analytics" size={24} color="#fff" />
                <Text style={styles.analyzeButtonText}>Analyze Shot</Text>
              </>
            )}
          </LinearGradient>
        </TouchableOpacity>

        {/* Results Section */}
        {result && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>Analysis Results</Text>

            {/* Intent Score Card */}
            <View style={styles.scoreCard}>
              <LinearGradient
                colors={[getScoreColor(result.intent_score), "#fff"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.scoreCardGradient}
              >
                <Text style={styles.scoreEmoji}>
                  {getFeedbackEmoji(result.intent_score)}
                </Text>
                <Text style={styles.scoreLabel}>Intent Execution Score</Text>
                <Text style={styles.scoreValue}>{result.intent_score}%</Text>
                <View
                  style={[
                    styles.correctBadge,
                    result.is_correct
                      ? styles.correctBadgeSuccess
                      : styles.correctBadgeError,
                  ]}
                >
                  <Text style={styles.correctBadgeText}>
                    {result.is_correct ? "✓ Correct Shot" : "✗ Different Shot"}
                  </Text>
                </View>
              </LinearGradient>
            </View>

            {/* Shot Details */}
            <View style={styles.detailsCard}>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Intended Shot:</Text>
                <Text style={styles.detailValue}>
                  {result.intended_shot.toUpperCase()}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Predicted Shot:</Text>
                <Text style={styles.detailValue}>
                  {result.predicted_shot.toUpperCase()}
                </Text>
              </View>
            </View>

            {/* Feedback */}
            <View style={styles.feedbackCard}>
              <Text style={styles.feedbackTitle}>Coach's Feedback</Text>
              <Text style={styles.feedbackText}>{result.feedback}</Text>
            </View>

            {/* Probability Distribution */}
            <View style={styles.probabilityCard}>
              <Text style={styles.probabilityTitle}>
                Shot Classification Confidence
              </Text>
              {Object.entries(result.probability_distribution)
                .sort(([, a], [, b]) => b - a)
                .map(([shot, probability]) => (
                  <View key={shot} style={styles.probabilityRow}>
                    <Text style={styles.probabilityLabel}>
                      {shot.toUpperCase()}
                    </Text>
                    <View style={styles.probabilityBarContainer}>
                      <View
                        style={[
                          styles.probabilityBar,
                          { width: `${probability * 100}%` },
                        ]}
                      />
                    </View>
                    <Text style={styles.probabilityValue}>
                      {(probability * 100).toFixed(1)}%
                    </Text>
                  </View>
                ))}
            </View>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f6fa",
  },
  header: {
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 20,
  },
  headerContent: {
    alignItems: "center",
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#fff",
    marginTop: 10,
  },
  headerSubtitle: {
    fontSize: 14,
    color: "#fff",
    opacity: 0.9,
    marginTop: 5,
  },
  content: {
    padding: 20,
  },
  section: {
    marginBottom: 25,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#1a237e",
    marginBottom: 5,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: "#7f8c8d",
    marginBottom: 15,
  },
  shotTypeGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  shotTypeButton: {
    backgroundColor: "#fff",
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 25,
    borderWidth: 2,
    borderColor: "#dfe6e9",
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
  },
  shotTypeButtonSelected: {
    backgroundColor: "#1a237e",
    borderColor: "#1a237e",
  },
  shotTypeText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#2c3e50",
  },
  shotTypeTextSelected: {
    color: "#fff",
  },
  uploadButton: {
    marginTop: 10,
    borderRadius: 15,
    overflow: "hidden",
  },
  uploadButtonGradient: {
    paddingVertical: 20,
    alignItems: "center",
    gap: 10,
  },
  uploadButtonText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#fff",
  },
  videoPreview: {
    marginTop: 15,
    alignItems: "center",
    gap: 10,
  },
  videoSelectedText: {
    fontSize: 16,
    color: "#2ecc71",
    fontWeight: "600",
  },
  analyzeButton: {
    marginVertical: 20,
    borderRadius: 15,
    overflow: "hidden",
  },
  analyzeButtonDisabled: {
    opacity: 0.6,
  },
  analyzeButtonGradient: {
    paddingVertical: 18,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  analyzeButtonText: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#fff",
  },
  resultsContainer: {
    marginTop: 20,
  },
  resultsTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1a237e",
    marginBottom: 20,
    textAlign: "center",
  },
  scoreCard: {
    borderRadius: 20,
    overflow: "hidden",
    marginBottom: 20,
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  scoreCardGradient: {
    padding: 30,
    alignItems: "center",
  },
  scoreEmoji: {
    fontSize: 50,
    marginBottom: 10,
  },
  scoreLabel: {
    fontSize: 16,
    color: "#2c3e50",
    marginBottom: 5,
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: "bold",
    color: "#1a237e",
  },
  correctBadge: {
    marginTop: 15,
    paddingVertical: 8,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  correctBadgeSuccess: {
    backgroundColor: "#2ecc71",
  },
  correctBadgeError: {
    backgroundColor: "#e74c3c",
  },
  correctBadgeText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 14,
  },
  detailsCard: {
    backgroundColor: "#fff",
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#ecf0f1",
  },
  detailLabel: {
    fontSize: 16,
    color: "#7f8c8d",
  },
  detailValue: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#1a237e",
  },
  feedbackCard: {
    backgroundColor: "#fff",
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    borderLeftWidth: 5,
    borderLeftColor: "#3498db",
  },
  feedbackTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#1a237e",
    marginBottom: 10,
  },
  feedbackText: {
    fontSize: 15,
    color: "#2c3e50",
    lineHeight: 22,
  },
  probabilityCard: {
    backgroundColor: "#fff",
    borderRadius: 15,
    padding: 20,
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1.41,
  },
  probabilityTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#1a237e",
    marginBottom: 15,
  },
  probabilityRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  probabilityLabel: {
    width: 70,
    fontSize: 13,
    fontWeight: "600",
    color: "#2c3e50",
  },
  probabilityBarContainer: {
    flex: 1,
    height: 20,
    backgroundColor: "#ecf0f1",
    borderRadius: 10,
    marginHorizontal: 10,
    overflow: "hidden",
  },
  probabilityBar: {
    height: "100%",
    backgroundColor: "#3498db",
    borderRadius: 10,
  },
  probabilityValue: {
    width: 50,
    fontSize: 13,
    fontWeight: "600",
    color: "#1a237e",
    textAlign: "right",
  },
});
