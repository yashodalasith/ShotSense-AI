import { Tabs } from "expo-router";
import React from "react";
import { Text, View, StyleSheet, Pressable, Animated } from "react-native";
import { useRouter, usePathname } from "expo-router";
import { Colors } from "@/constants/theme";
import { useColorScheme } from "@/hooks/use-color-scheme";

function CustomTabBar() {
  const router = useRouter();
  const pathname = usePathname();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === "dark";

  const [scaleValues] = React.useState({
    shot: new Animated.Value(1),
    home: new Animated.Value(1),
    stance_consistency: new Animated.Value(1),
  });

  const [glowValues] = React.useState({
    shot: new Animated.Value(0),
    home: new Animated.Value(0),
    stance_consistency: new Animated.Value(1),
  });

  const animateButton = (key: keyof typeof scaleValues) => {
    // Scale animation
    Animated.sequence([
      Animated.timing(scaleValues[key], {
        toValue: 0.85,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(scaleValues[key], {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start();

    // Glow pulse animation
    Animated.sequence([
      Animated.timing(glowValues[key], {
        toValue: 1,
        duration: 200,
        useNativeDriver: false,
      }),
      Animated.timing(glowValues[key], {
        toValue: 0,
        duration: 400,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const handlePress = (route: string, key: keyof typeof scaleValues) => {
    animateButton(key);
    setTimeout(() => router.push(route as any), 150);
  };

  const isActive = (route: string) => pathname.includes(route);

  const getGlowStyle = (key: keyof typeof glowValues, active: boolean) => {
    const glowColor = isDark
      ? "rgba(33, 150, 243, 0.4)"
      : "rgba(26, 35, 126, 0.3)";
    return {
      shadowColor: glowColor,
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: glowValues[key],
      shadowRadius: glowValues[key].interpolate({
        inputRange: [0, 1],
        outputRange: [0, 12],
      }),
      elevation: glowValues[key].interpolate({
        inputRange: [0, 1],
        outputRange: [0, 8],
      }),
    };
  };

  return (
    <View style={[styles.tabBar, isDark && styles.tabBarDark]}>
      {/* Shot Classification */}
      <Animated.View
        style={{
          transform: [{ scale: scaleValues.shot }],
        }}
      >
        <Pressable
          style={styles.tabButton}
          onPress={() =>
            handlePress("/shotclassify/ShotClassification", "shot")
          }
        >
          <Animated.View
            style={[
              styles.iconContainer,
              isActive("shotclassify") && styles.iconContainerActive,
              getGlowStyle("shot", isActive("shotclassify")),
            ]}
          >
            <Text
              style={[
                styles.icon,
                isActive("shotclassify") && styles.iconActive,
              ]}
            >
              🏏
            </Text>
            {isActive("shotclassify") && (
              <View
                style={[styles.activeDot, isDark && styles.activeDotDark]}
              />
            )}
          </Animated.View>
          <Text
            style={[
              styles.label,
              isDark && styles.labelDark,
              isActive("shotclassify") && styles.labelActive,
              isActive("shotclassify") && isDark && styles.labelActiveDark,
            ]}
          >
            Shot Classify
          </Text>
        </Pressable>
      </Animated.View>

      {/* Home Button - Center Elevated */}
      <Animated.View
        style={[
          styles.homeButtonContainer,
          { transform: [{ scale: scaleValues.home }] },
        ]}
      >
        <Animated.View
          style={getGlowStyle("home", isActive("/") || pathname === "/")}
        >
          <Pressable
            style={[styles.homeButton, isDark && styles.homeButtonDark]}
            onPress={() => handlePress("/", "home")}
          >
            <Text style={styles.homeIcon}>🏏</Text>
            {(isActive("/") || pathname === "/") && (
              <View
                style={[
                  styles.homeActiveDot,
                  isDark && styles.homeActiveDotDark,
                ]}
              />
            )}
          </Pressable>
        </Animated.View>
        <Text
          style={[
            styles.homeLabel,
            isDark && styles.homeLabelDark,
            (isActive("/") || pathname === "/") && styles.homeLabelActive,
            (isActive("/") || pathname === "/") &&
              isDark &&
              styles.homeLabelActiveDark,
          ]}
        >
          Home
        </Text>
      </Animated.View>

      {/* Consistency */}
      <Animated.View
        style={{
          transform: [{ scale: scaleValues.stance_consistency }],
        }}
      >
        <Pressable
          style={styles.tabButton}
          onPress={() =>
            handlePress(
              "/stanceconsistency/StanceConsistency",
              "stance_consistency"
            )
          }
        >
          <Animated.View
            style={[
              styles.iconContainer,
              isActive("stance_consistency") && styles.iconContainerActive,
              getGlowStyle(
                "stance_consistency",
                isActive("stance_consistency")
              ),
            ]}
          >
            <Text
              style={[
                styles.icon,
                isActive("stance_consistency") && styles.iconActive,
              ]}
            >
              ⚖️
            </Text>
            {isActive("stance_consistency") && (
              <View
                style={[styles.activeDot, isDark && styles.activeDotDark]}
              />
            )}
          </Animated.View>
          <Text
            style={[
              styles.label,
              isDark && styles.labelDark,
              isActive("prediction") && styles.labelActive,
              isActive("prediction") && isDark && styles.labelActiveDark,
            ]}
          >
            Stance Consistency
          </Text>
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    flexDirection: "row",
    backgroundColor: "#ffffff",
    borderTopWidth: 1,
    borderTopColor: "#e0e0e0",
    paddingVertical: 8,
    paddingHorizontal: 8,
    height: 75,
    alignItems: "center",
    justifyContent: "space-around",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: -3 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 12,
  },
  tabBarDark: {
    backgroundColor: "#1a1a2e",
    borderTopColor: "#2a2a3e",
  },
  tabButton: {
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "transparent",
    marginBottom: 4,
    position: "relative",
  },
  iconContainerActive: {
    backgroundColor: "rgba(26, 35, 126, 0.1)",
  },
  icon: {
    fontSize: 24,
  },
  iconActive: {
    fontSize: 26,
  },
  activeDot: {
    position: "absolute",
    bottom: 2,
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#1a237e",
  },
  activeDotDark: {
    backgroundColor: "#2196F3",
  },
  label: {
    fontSize: 10,
    fontWeight: "600",
    color: "#666",
    marginTop: 2,
  },
  labelDark: {
    color: "#999",
  },
  labelActive: {
    color: "#1a237e",
    fontWeight: "700",
  },
  labelActiveDark: {
    color: "#2196F3",
  },
  homeButtonContainer: {
    alignItems: "center",
    marginTop: -30,
    marginHorizontal: 8,
  },
  homeButton: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "#1a237e",
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#1a237e",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 8,
    borderWidth: 4,
    borderColor: "#ffffff",
    position: "relative",
  },
  homeButtonDark: {
    backgroundColor: "#2196F3",
    borderColor: "#1a1a2e",
    shadowColor: "#2196F3",
  },
  homeIcon: {
    fontSize: 32,
  },
  homeActiveDot: {
    position: "absolute",
    bottom: 8,
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "#ffffff",
  },
  homeActiveDotDark: {
    backgroundColor: "#1a1a2e",
  },
  homeLabel: {
    fontSize: 10,
    fontWeight: "700",
    color: "#1a237e",
    marginTop: 4,
  },
  homeLabelDark: {
    color: "#2196F3",
  },
  homeLabelActive: {
    color: "#1a237e",
    fontWeight: "800",
  },
  homeLabelActiveDark: {
    color: "#2196F3",
  },
});

export default function TabLayout() {
  const colorScheme = useColorScheme();

  return (
    <Tabs
      tabBar={() => <CustomTabBar />}
      screenOptions={{
        headerShown: false,
      }}
    >
      <Tabs.Screen name="index" />
      <Tabs.Screen name="shotclassify/ShotClassification" />
      <Tabs.Screen name="shotsimilarity/ShotSimilarity" />
      <Tabs.Screen name="legalitycheck/BowlingCheck" />
      <Tabs.Screen name="prediction/Prediction" />
    </Tabs>
  );
}
