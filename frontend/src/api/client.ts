export type HealthResponse = {
  status: string;
  app: string;
  env: string;
  mock: boolean;
};

export type SlideMetadata = {
  slide_id: string;
  slide_path: string;
  width: number;
  height: number;
  level_count: number;
  level_dimensions: Array<[number, number]>;
  mpp_x: number | null;
  mpp_y: number | null;
  objective_power: number | null;
  stain: string;
  species: string;
  organ: string;
  mock: boolean;
};

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error("Health request failed");
  }
  return response.json();
}

export async function registerSlide(slidePath: string): Promise<SlideMetadata> {
  const response = await fetch("/api/slides/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      slide_path: slidePath,
      species: "rat",
      organ: "liver",
      stain: "H&E",
    }),
  });
  if (!response.ok) {
    throw new Error("Slide registration failed");
  }
  return response.json();
}

