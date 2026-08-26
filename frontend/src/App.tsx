import { CommandCenter } from "./features/command-center/CommandCenter";
import { DiagnosticsPage } from "./features/diagnostics/DiagnosticsPage";

function App() {
  const path = window.location.pathname.replace(/\/$/, "") || "/";
  const showDiagnostics = path === "/system" || path === "/diagnostics";

  return showDiagnostics ? <DiagnosticsPage /> : <CommandCenter />;
}

export default App;
