import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // 競馬枠番カラー
        frame: {
          1: "#ffffff", // 白
          2: "#1a1a1a", // 黒
          3: "#dc2626", // 赤
          4: "#2563eb", // 青
          5: "#fbbf24", // 黄
          6: "#16a34a", // 緑
          7: "#f97316", // 橙
          8: "#ec4899", // 桃
        },
      },
    },
  },
  plugins: [],
};

export default config;
