import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Autoresearch Dashboard',
  description: 'Vol regime framework diagnostics and insights',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
