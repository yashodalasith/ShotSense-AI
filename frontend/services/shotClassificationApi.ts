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
  severity: "critical" | "major" | "minor" | "negligible"; // ✅ Added "negligible"
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
  joint_connections?: Array<{
    from: string;
    to: string;
    label: string;
  }>;
  legacy_images?: {
    skeleton_3d?: string;
    comparison_view?: string;
    animation_360?: string;
  };
  prototype_used?: string;
  prototype_samples?: number;
}

export interface AnalysisResult {
  intended_shot: string;
  predicted_shot: string;
  intent_score: number;
  is_correct: boolean;
  visual_feedback: VisualFeedback;
  coaching_feedback: string;
  ensemble_probabilities: Record<string, number>;

  // ✅ Added missing fields
  mistake_analysis?: Array<{
    body_part: string;
    joint_id: string;
    feature_name: string;
    severity: string;
    severity_score: number;
    actual_value: number;
    expected_value: number;
    deviation: number;
    importance: number;
    explanation: string;
    recommendation: string;
  }>;

  correction_summary?: string;

  model_predictions?: {
    random_forest: string;
    xgboost: string;
    gradient_boosting: string;
  };

  analysis_metadata: {
    contact_frame: number; // ✅ Made required
    contact_detection?: {
      contact_frame?: number;
      detection_method?: string;
      ball_detected?: boolean;
      bat_detected?: boolean;
      virtual_bat_used?: boolean;
      ball_detection_rate?: number;
      bat_detection_rate?: number;
      tier1_score?: number;
      tier2_score?: number;
    };
    prototype_samples?: number;
    analysis_method?: string;
  };
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
  intendedShot: string,
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
    body: formData,
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

// -------------------------
// Stance Consistency APIs
// -------------------------

export interface StanceQuickCompareResult {
  similarity_score: number;
  rating: string;
  video1_consistency: number;
  video2_consistency: number;
  feedback: string;
}

export interface StanceAnalysisResult {
  summary: any;
  individual_video_scores: any[];
  consistency_analysis: any;
  feedback: any;
  insights?: any;
}

/**
 * Quick compare two stance videos
 */
export const quickCompareStances = async (
  videoUri1: string,
  videoUri2: string,
): Promise<StanceQuickCompareResult> => {
  const formData = new FormData();

  if (Platform.OS === "web") {
    const r1 = await fetch(videoUri1);
    const b1 = await r1.blob();
    formData.append("video1", b1, "video1.mp4");
    const r2 = await fetch(videoUri2);
    const b2 = await r2.blob();
    formData.append("video2", b2, "video2.mp4");
  } else {
    const n1 = videoUri1.split("/").pop() || "video1.mp4";
    const n2 = videoUri2.split("/").pop() || "video2.mp4";
    formData.append("video1", {
      uri: videoUri1,
      name: n1,
      type: "video/mp4",
    } as any);
    formData.append("video2", {
      uri: videoUri2,
      name: n2,
      type: "video/mp4",
    } as any);
  }

  const response = await fetch(
    `${API_BASE_URL}/stance-consistency/quick-compare`,
    {
      method: "POST",
      body: formData,
    },
  );

  const data = await response.json();

  if (data.success && data.data) {
    return data.data as StanceQuickCompareResult;
  }

  throw new Error(data.message || "Quick compare failed");
};

/**
 * Analyze multiple stance videos (full consistency analysis)
 */
export const analyzeStanceConsistency = async (
  videoUris: string[],
): Promise<StanceAnalysisResult> => {
  const formData = new FormData();

  if (!Array.isArray(videoUris) || videoUris.length < 2) {
    throw new Error(
      "At least two videos are required for stance consistency analysis",
    );
  }

  if (Platform.OS === "web") {
    for (let i = 0; i < videoUris.length; i++) {
      const r = await fetch(videoUris[i]);
      const b = await r.blob();
      formData.append("videos", b, `video_${i + 1}.mp4`);
    }
  } else {
    for (let i = 0; i < videoUris.length; i++) {
      const uri = videoUris[i];
      const name = uri.split("/").pop() || `video_${i + 1}.mp4`;
      formData.append("videos", { uri, name, type: "video/mp4" } as any);
    }
  }

  const response = await fetch(`${API_BASE_URL}/stance-consistency/analyze`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (data.success && data.data) {
    return data.data as StanceAnalysisResult;
  }

  throw new Error(data.message || "Stance consistency analysis failed");
};
