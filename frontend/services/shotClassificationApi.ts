/**
 * Shot Classification API Service
 * Handles communication with FastAPI backend
 */
import { Platform } from "react-native";
const API_BASE_URL =
  Platform.OS === "android"
    ? process.env.EXPO_PUBLIC_BATTING_SHOT_API_ANDROID
    : process.env.EXPO_PUBLIC_BATTING_SHOT_API_LOCAL;

export interface ShotType {
  value: string;
  label: string;
}

export interface Keypoint3D {
  joint: string;
  index: number;
  position: { x: number; y: number; z: number };
}

export interface Mistake {
  joint_id: string;
  body_part: string;
  severity: "critical" | "major" | "minor";
  severity_color: string;
  glow_intensity: number;
  explanation: string;
  recommendation: string;
}

export interface VisualFeedback {
  keypoints_3d: {
    actual: Keypoint3D[];
    prototype: Keypoint3D[];
    format?: string;
  };
  mistakes: Mistake[];
  // other fields like legacy_images, etc.
}

export interface AnalysisResult {
  intended_shot: string;
  predicted_shot: string;
  intent_score: number;
  is_correct: boolean;
  visual_feedback: VisualFeedback;
  coaching_feedback: string;
  ensemble_probabilities: Record<string, number>;
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
