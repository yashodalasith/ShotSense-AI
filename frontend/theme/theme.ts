export const theme = {
  colors: {
    background: "#0D1211",
    text: "#000000",
    subtext: "#666666",
    accent: "#007AFF", // Standard accessible blue or brand color
    card: "#FFFFFF", // Standard card background
    cardBorder: "#E5E5EA",
    error: "#FF3B30",
    success: "#34C759",
    // Dark theme colors for bottom nav
    navBackground: "#1E1E1E", // Dark gray background
    navCard: "#2A2A2A", // Dark gray card
    navBorder: "rgba(255, 255, 255, 0.1)",
    neonAccent: "#00FF9D", // Neon green accent
    navActiveText: "#FFFFFF",
    navInactiveText: "rgba(255, 255, 255, 0.3)",
  },
  spacing: {
    s: 8,
    m: 16,
    l: 24,
    xl: 32,
  },
} as const;
