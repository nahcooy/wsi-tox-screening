import { useEffect, useState } from "react";
import { Activity, FileImage, Play } from "lucide-react";
import { getHealth, registerSlide, type HealthResponse, type SlideMetadata } from "../api/client";

export function WorkbenchPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [slidePath, setSlidePath] = useState("/data/mock_rat_liver.svs");
  const [metadata, setMetadata] = useState<SlideMetadata | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  async function handleRegisterSlide() {
    const result = await registerSlide(slidePath);
    setMetadata(result);
  }

  return (
    <main className="workbench">
      <header className="topbar">
        <strong>WSI Toxicity Screening</strong>
        <span className={health?.status === "ok" ? "status ok" : "status"}>
          <Activity size={14} />
          {health?.status ?? "offline"}
        </span>
      </header>

      <section className="grid">
        <aside className="panel">
          <h1>Configuration</h1>
          <label>
            Slide path
            <input value={slidePath} onChange={(event) => setSlidePath(event.target.value)} />
          </label>
          <button onClick={handleRegisterSlide}>
            <FileImage size={16} />
            Load Mock Slide
          </button>
          <button disabled>
            <Play size={16} />
            Run MIL
          </button>
        </aside>

        <section className="viewer">
          <div className="viewerSurface">
            <span>OpenSeadragon viewer placeholder</span>
          </div>
        </section>

        <aside className="panel">
          <h2>MIL Result</h2>
          <p>Mock mode is ready. Real inference will be added in later tasks.</p>
          <h2>Slide Metadata</h2>
          <pre>{JSON.stringify(metadata ?? {}, null, 2)}</pre>
        </aside>
      </section>
    </main>
  );
}

