/**
 * Shot Classification API Service
 * Handles communication with FastAPI backend
 */
import { Platform } from "react-native";

// Update this with your actual API URL
const API_BASE_URL = "http://localhost:8001"; // Change to your backend URL

export interface ShotType {
  value: string;
  label: string;
}

export interface AnalysisResult {
  intended_shot: string;
  predicted_shot: string;
  intent_score: number;
  probability_distribution: Record<string, number>;
  feedback: string;
  is_correct: boolean;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message: string;
}

/**
 * Get available shot types from API
 */
export const getShotTypes = async (): Promise<ShotType[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/batting/shot-types`);
    const data = await response.json();

    if (data.success && Array.isArray(data.shot_types)) {
      return data.shot_types.map((type: string) => ({
        value: type,
        label: type.charAt(0).toUpperCase() + type.slice(1),
      }));
    }

    throw new Error(data.message || "Failed to fetch shot types");
  } catch (error) {
    console.error("Error fetching shot types:", error);
    throw error;
  }
};

/**
 * Analyze cricket shot video
 */

export const analyzeShot = async (
  videoUri: string,
  intendedShot: string
): Promise<AnalysisResult> => {
  const formData = new FormData();

  if (Platform.OS === "web") {
    // WEB: convert URI → Blob
    const response = await fetch(videoUri);
    const blob = await response.blob();

    formData.append("video", blob, "video.mp4");
  } else {
    // MOBILE: use file object
    const filename = videoUri.split("/").pop() || "video.mp4";

    formData.append("video", {
      uri: videoUri,
      name: filename,
      type: "video/mp4",
    } as any);
  }

  formData.append("intended_shot", intendedShot);

  const response = await fetch(`${API_BASE_URL}/batting/analyze-shot`, {
    method: "POST",
    body: formData, // ❗ no headers
  });

  const data = await response.json();

  if (data.success && data.data) {
    return data.data;
  }

  throw new Error(data.message || "Analysis failed");
};

/**
 * Check API health
 */
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/batting/health`);
    const data = await response.json();
    return data.success;
  } catch (error) {
    console.error("API health check failed:", error);
    return false;
  }
};
