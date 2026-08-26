import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("FloodSight interface error", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="grid min-h-screen place-items-center bg-[#071016] p-6 text-slate-100">
          <section className="max-w-md rounded-2xl border border-rose-400/20 bg-rose-400/[0.06] p-8 text-center">
            <p className="text-xs font-bold tracking-[0.2em] text-rose-300 uppercase">Interface fault</p>
            <h1 className="mt-3 text-2xl font-semibold">FloodSight could not render this view.</h1>
            <p className="mt-3 text-sm leading-6 text-slate-400">Reload the page. If the issue continues, run the frontend checks and inspect the browser console.</p>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
}

