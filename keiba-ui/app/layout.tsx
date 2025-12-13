import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "競馬シナリオ分析UI",
  description: "競馬予想AIのシナリオ補正・分析ツール",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body className="min-h-screen bg-gray-50">{children}</body>
    </html>
  );
}
