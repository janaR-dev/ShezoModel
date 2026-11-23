/** Tailwind build for the Shezo web app (output: web/tailwind.css). */
module.exports = {
  content: [require("path").join(__dirname, "index.html"), require("path").join(__dirname, "app.js")],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      colors: {
        ink: {
          950: "#070b14",
          900: "#0b1220",
          850: "#101a2c",
          800: "#16223a",
        },
        mint: {
          300: "#7ff2d0",
          400: "#3ee0b4",
          500: "#1cc497",
        },
      },
      boxShadow: {
        panel: "0 24px 60px -30px rgba(3, 12, 28, 0.9)",
        glow: "0 0 0 1px rgba(62, 224, 180, 0.25), 0 18px 40px -24px rgba(62, 224, 180, 0.6)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: { rise: "rise 260ms ease-out both" },
    },
  },
  plugins: [],
};
