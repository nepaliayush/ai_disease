import "./globals.css";

import { HeartPulse } from "lucide-react";
import Link from "next/link";

import Navbar from "@/components/Navbar";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";

export const metadata = {
  title: "Multi-Model Disease Risk Assessment",
  description:
    "Fuses four clinical ML models (diabetes, heart, liver, CKD) with a symptom triage signal using decision-level fusion.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <header className="mx-auto flex h-14 max-w-[1600px] items-center justify-between gap-4 border-b px-6">
            <Link href="/" className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground">
                <HeartPulse className="h-4 w-4" />
              </span>
              <span className="text-sm font-semibold">
                Multi-Model Risk Assessment
              </span>
            </Link>
            <Navbar />
            <ThemeToggle />
          </header>
          <main className="mx-auto max-w-[1600px] px-6 py-8">{children}</main>
          <footer className="mx-auto max-w-[1600px] px-6 pb-8 text-center text-xs text-muted-foreground">
            {/* <p>
              Built with Next.js + shadcn/ui (frontend) and FastAPI (backend).
              Public, anonymized datasets only. For research and education — not
              a medical device.
            </p> */}
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}