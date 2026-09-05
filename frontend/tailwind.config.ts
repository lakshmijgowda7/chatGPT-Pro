import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        sidebar: {
          DEFAULT: "#171717",
          hover: "#212121",
        },
        chat: {
          bubble: "#2f2f2f",
          input: "#212121",
        },
        brand: {
          DEFAULT: "#10a37f",
          hover: "#1a7f64",
        },
      },
    },
  },
  plugins: [],
};

export default config;
