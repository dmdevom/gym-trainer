import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "trAIner · AI workout analysis",
  description: "Upload or record a workout set and get AI-powered rep counting, form feedback, and landmarked video analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body suppressHydrationWarning>{children}</body></html>;
}
