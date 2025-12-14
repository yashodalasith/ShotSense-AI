import React from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
} from "react-native";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";

export default function HomeScreen() {
  const router = useRouter();
  const fadeAnim = new Animated.Value(0);

  React.useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();
  }, []);

  const features = [
    {
      id: 1,
      icon: "🏏",
      title: "Shot Classification",
      description:
        "Identify cricket shots like pull, cover drive, straight drive and more",
      route: "/shotclassify/ShotClassification" as const,
      gradient: ["#f093fb", "#f5576c"],
    },
  ] as const;

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header with Gradient */}
      <LinearGradient
        colors={["#667eea", "#764ba2"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.header}
      >
        <Animated.View style={[styles.headerContent, { opacity: fadeAnim }]}>
          <Text style={styles.logo}>🏏</Text>
          <Text style={styles.headerTitle}>Cricket AI Coach</Text>
          <Text style={styles.headerSubtitle}>
            AI-Powered Cricket Practice & Analysis Platform
          </Text>
          <View style={styles.headerBadge}>
            <Text style={styles.badgeText}>✨ Premium Training</Text>
          </View>
        </Animated.View>
      </LinearGradient>

      {/* Hero Section with Card Style */}
      <View style={styles.heroSection}>
        <View style={styles.heroCard}>
          <Text style={styles.heroTitle}>
            Master Your Cricket Skills with AI
          </Text>
          <Text style={styles.heroDescription}>
            Advanced machine learning technology to analyze bowling actions,
            classify shots, compare techniques, and predict match outcomes
          </Text>
          <View style={styles.heroStats}>
            <View style={styles.heroStatItem}>
              <Text style={styles.heroStatNumber}>10K+</Text>
              <Text style={styles.heroStatLabel}>Users</Text>
            </View>
            <View style={styles.heroStatDivider} />
            <View style={styles.heroStatItem}>
              <Text style={styles.heroStatNumber}>95%</Text>
              <Text style={styles.heroStatLabel}>Accuracy</Text>
            </View>
            <View style={styles.heroStatDivider} />
            <View style={styles.heroStatItem}>
              <Text style={styles.heroStatNumber}>24/7</Text>
              <Text style={styles.heroStatLabel}>Available</Text>
            </View>
          </View>
        </View>
      </View>

      {/* Features Grid with Gradient Cards */}
      <View style={styles.featuresSection}>
        <Text style={styles.sectionTitle}>Choose Your Training Module</Text>
        <Text style={styles.sectionSubtitle}>
          Select from our AI-powered modules
        </Text>

        {features.map((feature, index) => (
          <Animated.View
            key={feature.id}
            style={[
              { opacity: fadeAnim },
              {
                transform: [
                  {
                    translateY: fadeAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [50, 0],
                    }),
                  },
                ],
              },
            ]}
          >
            <TouchableOpacity
              style={styles.featureCardWrapper}
              onPress={() => router.push(feature.route)}
              activeOpacity={0.9}
            >
              <LinearGradient
                colors={feature.gradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.featureCard}
              >
                <View style={styles.featureIconContainer}>
                  <Text style={styles.featureIcon}>{feature.icon}</Text>
                </View>
                <View style={styles.featureContent}>
                  <Text style={styles.featureTitle}>{feature.title}</Text>
                  <Text style={styles.featureDescription}>
                    {feature.description}
                  </Text>
                </View>
                <View style={styles.featureArrowContainer}>
                  <Text style={styles.featureArrow}>→</Text>
                </View>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>
        ))}
      </View>

      {/* How It Works with Modern Design */}
      <View style={styles.howItWorksSection}>
        <Text style={styles.sectionTitle}>How It Works</Text>
        <Text style={styles.sectionSubtitle}>
          Three simple steps to improve your game
        </Text>

        <View style={styles.stepsContainer}>
          {[
            {
              num: "1",
              title: "Select Module",
              desc: "Choose from 4 AI-powered training modules",
              icon: "📱",
            },
            {
              num: "2",
              title: "Upload Media",
              desc: "Provide images or videos for analysis",
              icon: "📸",
            },
            {
              num: "3",
              title: "Get Results",
              desc: "Receive detailed AI-powered insights",
              icon: "✨",
            },
          ].map((step, index) => (
            <View key={step.num}>
              <View style={styles.stepItem}>
                <View style={styles.stepIconBg}>
                  <Text style={styles.stepIcon}>{step.icon}</Text>
                </View>
                <View style={styles.stepContent}>
                  <View style={styles.stepHeader}>
                    <View style={styles.stepNumberBadge}>
                      <Text style={styles.stepNumber}>{step.num}</Text>
                    </View>
                    <Text style={styles.stepTitle}>{step.title}</Text>
                  </View>
                  <Text style={styles.stepDescription}>{step.desc}</Text>
                </View>
              </View>
              {index < 2 && (
                <View style={styles.stepConnector}>
                  <View style={styles.stepConnectorLine} />
                </View>
              )}
            </View>
          ))}
        </View>
      </View>

      {/* CTA Section with Gradient */}
      <LinearGradient
        colors={["#667eea", "#764ba2"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.ctaSection}
      >
        <Text style={styles.ctaTitle}>Ready to Improve Your Game?</Text>
        <Text style={styles.ctaDescription}>
          Start using our AI-powered modules and take your cricket skills to the
          next level
        </Text>
        <TouchableOpacity style={styles.ctaButton} activeOpacity={0.8}>
          <Text style={styles.ctaButtonText}>Get Started Now</Text>
          <Text style={styles.ctaButtonArrow}>→</Text>
        </TouchableOpacity>
      </LinearGradient>

      {/* Footer with Modern Design */}
      <View style={styles.footer}>
        <View style={styles.footerContent}>
          <Text style={styles.footerLogo}>🏏</Text>
          <Text style={styles.footerTitle}>Cricket AI Coach</Text>
          <Text style={styles.footerTagline}>
            Powered by Advanced Machine Learning
          </Text>

          <View style={styles.footerLinks}>
            <TouchableOpacity style={styles.footerLink}>
              <Text style={styles.footerLinkText}>About</Text>
            </TouchableOpacity>
            <Text style={styles.footerDivider}>•</Text>
            <TouchableOpacity style={styles.footerLink}>
              <Text style={styles.footerLinkText}>Privacy</Text>
            </TouchableOpacity>
            <Text style={styles.footerDivider}>•</Text>
            <TouchableOpacity style={styles.footerLink}>
              <Text style={styles.footerLinkText}>Contact</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.footerModules}>
            <Text style={styles.footerModulesTitle}>Available Modules</Text>
            <Text style={styles.footerModulesText}>Shot Classification</Text>
          </View>

          <Text style={styles.footerCopyright}>
            © 2024 Cricket AI Coach. All rights reserved.
          </Text>
          <Text style={styles.footerVersion}>Version 1.0.0</Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8f9fd",
  },
  header: {
    paddingTop: 60,
    paddingBottom: 50,
    paddingHorizontal: 20,
  },
  headerContent: {
    alignItems: "center",
  },
  logo: {
    fontSize: 70,
    marginBottom: 16,
    textShadowColor: "rgba(0, 0, 0, 0.2)",
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },
  headerTitle: {
    fontSize: 34,
    fontWeight: "bold",
    color: "#fff",
    marginBottom: 8,
    textAlign: "center",
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 15,
    color: "rgba(255, 255, 255, 0.9)",
    textAlign: "center",
    paddingHorizontal: 20,
    lineHeight: 22,
  },
  headerBadge: {
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.3)",
  },
  badgeText: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "600",
  },
  heroSection: {
    paddingHorizontal: 20,
    marginTop: -30,
    marginBottom: 20,
  },
  heroCard: {
    backgroundColor: "#fff",
    borderRadius: 24,
    padding: 28,
    shadowColor: "#667eea",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 16,
    elevation: 8,
  },
  heroTitle: {
    fontSize: 26,
    fontWeight: "bold",
    color: "#1a1a2e",
    textAlign: "center",
    marginBottom: 12,
    letterSpacing: 0.3,
  },
  heroDescription: {
    fontSize: 15,
    color: "#666",
    textAlign: "center",
    lineHeight: 24,
    marginBottom: 24,
  },
  heroStats: {
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
  },
  heroStatItem: {
    alignItems: "center",
    flex: 1,
  },
  heroStatNumber: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#667eea",
    marginBottom: 4,
  },
  heroStatLabel: {
    fontSize: 12,
    color: "#999",
    fontWeight: "500",
  },
  heroStatDivider: {
    width: 1,
    height: 30,
    backgroundColor: "#e0e0e0",
  },
  featuresSection: {
    paddingHorizontal: 20,
    paddingVertical: 30,
  },
  sectionTitle: {
    fontSize: 26,
    fontWeight: "bold",
    color: "#1a1a2e",
    marginBottom: 8,
    textAlign: "center",
    letterSpacing: 0.3,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: "#999",
    textAlign: "center",
    marginBottom: 28,
  },
  featureCardWrapper: {
    marginBottom: 16,
  },
  featureCard: {
    flexDirection: "row",
    alignItems: "center",
    padding: 20,
    borderRadius: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 5,
  },
  featureIconContainer: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 16,
  },
  featureIcon: {
    fontSize: 32,
  },
  featureContent: {
    flex: 1,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#fff",
    marginBottom: 6,
    letterSpacing: 0.2,
  },
  featureDescription: {
    fontSize: 13,
    color: "rgba(255, 255, 255, 0.9)",
    lineHeight: 19,
  },
  featureArrowContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 8,
  },
  featureArrow: {
    fontSize: 20,
    color: "#fff",
    fontWeight: "bold",
  },
  howItWorksSection: {
    paddingHorizontal: 20,
    paddingVertical: 30,
    backgroundColor: "#fff",
  },
  stepsContainer: {
    marginTop: 10,
  },
  stepItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f8f9fd",
    padding: 20,
    borderRadius: 16,
    marginBottom: 8,
  },
  stepIconBg: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 16,
    shadowColor: "#667eea",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  stepIcon: {
    fontSize: 28,
  },
  stepContent: {
    flex: 1,
  },
  stepHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  stepNumberBadge: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "#667eea",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 10,
  },
  stepNumber: {
    fontSize: 14,
    fontWeight: "bold",
    color: "#fff",
  },
  stepTitle: {
    fontSize: 17,
    fontWeight: "bold",
    color: "#1a1a2e",
    letterSpacing: 0.2,
  },
  stepDescription: {
    fontSize: 13,
    color: "#666",
    lineHeight: 19,
  },
  stepConnector: {
    alignItems: "center",
    paddingVertical: 8,
  },
  stepConnectorLine: {
    width: 2,
    height: 20,
    backgroundColor: "#e0e0e0",
  },
  ctaSection: {
    paddingHorizontal: 20,
    paddingVertical: 50,
    alignItems: "center",
    marginTop: 20,
  },
  ctaTitle: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#fff",
    marginBottom: 12,
    textAlign: "center",
    letterSpacing: 0.3,
  },
  ctaDescription: {
    fontSize: 15,
    color: "rgba(255, 255, 255, 0.9)",
    textAlign: "center",
    lineHeight: 24,
    marginBottom: 28,
    paddingHorizontal: 10,
  },
  ctaButton: {
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 30,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  ctaButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#667eea",
    marginRight: 8,
    letterSpacing: 0.3,
  },
  ctaButtonArrow: {
    fontSize: 20,
    color: "#667eea",
    fontWeight: "bold",
  },
  footer: {
    paddingHorizontal: 20,
    paddingTop: 50,
    paddingBottom: 40,
    backgroundColor: "#1a1a2e",
  },
  footerContent: {
    alignItems: "center",
  },
  footerLogo: {
    fontSize: 56,
    marginBottom: 16,
  },
  footerTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#fff",
    marginBottom: 8,
    letterSpacing: 0.5,
  },
  footerTagline: {
    fontSize: 14,
    color: "rgba(255, 255, 255, 0.7)",
    marginBottom: 32,
  },
  footerLinks: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 32,
  },
  footerLink: {
    paddingHorizontal: 16,
  },
  footerLinkText: {
    fontSize: 14,
    color: "rgba(255, 255, 255, 0.8)",
    fontWeight: "500",
  },
  footerDivider: {
    fontSize: 14,
    color: "rgba(255, 255, 255, 0.3)",
  },
  footerModules: {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    padding: 20,
    borderRadius: 16,
    marginBottom: 32,
    width: "100%",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  footerModulesTitle: {
    fontSize: 15,
    fontWeight: "bold",
    color: "#fff",
    marginBottom: 10,
    textAlign: "center",
    letterSpacing: 0.5,
  },
  footerModulesText: {
    fontSize: 12,
    color: "rgba(255, 255, 255, 0.7)",
    textAlign: "center",
    lineHeight: 20,
  },
  footerCopyright: {
    fontSize: 12,
    color: "rgba(255, 255, 255, 0.5)",
    marginBottom: 6,
    textAlign: "center",
  },
  footerVersion: {
    fontSize: 11,
    color: "rgba(255, 255, 255, 0.4)",
    textAlign: "center",
  },
});
