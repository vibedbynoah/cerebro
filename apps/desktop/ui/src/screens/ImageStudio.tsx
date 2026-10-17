import "../styles/images.css";
import { createMemo, createSignal, For, onCleanup, Show, type JSX } from "solid-js";
import { Icon } from "../components/icons";
import { IllustrativeFlag, ScreenHeader } from "../components/Screen";
import { Preview } from "../components/Preview";
import {
  ASPECTS,
  estimateSeconds,
  hashPrompt,
  IMAGE_MODELS,
  loadGallery,
  MODE_LABELS,
  MODES,
  randomSeed,
  rng,
  SAMPLERS,
  saveGallery,
  sizeFor,
  validate,
  type GalleryItem,
  type GenParams,
} from "../lib/images";
import { newId } from "../lib/chat/store";
import { notify } from "../lib/toast";
import { useAppState } from "../state";

/**
 * Image Studio: native image generation on this machine. Text to image, image to image,
 * inpainting with a brush mask, upscaling and variations; local models (FLUX, SD 3.5, SDXL,
 * Qwen-Image, GGUF image models); live denoising preview; a gallery with favourites, prompt reuse
 * and export. In this build images are illustrative artwork drawn from the seed.
 */
export function ImageStudio(): JSX.Element {
  // The desktop engine runs language models only; it says so instead of drawing stand-in artwork.
  if (__DESKTOP__) return <ImagesPlanned />;
  return <ImageStudioPreview />;
}

function ImagesPlanned(): JSX.Element {
  return (
    <div class="screen">
      <ScreenHeader route="images" />
      <Preview
        art="gallery"
        title="Images made on this computer"
        lede="Cerebro Engine runs language models today. Image models come next, on the same GPU and with the same speed tuning."
        features={[
          { icon: "images", title: "Text to image", text: "Describe a picture; it is drawn locally with the seed shown so you can make it again." },
          { icon: "sliders", title: "Edit and upscale", text: "Paint a mask to change part of a picture, or upscale it, without uploading anything." },
          { icon: "gpu", title: "Fits your card", text: "Picks the image model and resolution your VRAM can hold." },
        ]}
      />
    </div>
  );
}

/** The browser preview's design study of the screen, with illustrative artwork drawn from the seed. */
function ImageStudioPreview(): JSX.Element {
  const app = useAppState();
  const [advanced, setAdvanced] = createSignal(app.level() === "advanced");
  const [p, setP] = createSignal<GenParams>({
    mode: "txt2img",
    modelId: IMAGE_MODELS[0]!.id,
    prompt: "",
    negative: "",
    aspect: "1:1",
    seed: null,
    steps: IMAGE_MODELS[0]!.defaultSteps,
    guidance: IMAGE_MODELS[0]!.defaultGuidance,
    sampler: "Euler",
    strength: 0.6,
    scale: 2,
  });
  const set = (patch: Partial<GenParams>) => setP((cur) => ({ ...cur, ...patch }));
  const [source, setSource] = createSignal<string | null>(null);
  const [hasMask, setHasMask] = createSignal(false);
  const [progress, setProgress] = createSignal<number | null>(null);
  const [current, setCurrent] = createSignal<GalleryItem | null>(null);
  const [gallery, setGallery] = createSignal<GalleryItem[]>(loadGallery());
  const [filter, setFilter] = createSignal<"all" | "favourites">("all");
  const [tried, setTried] = createSignal(false);
  let timer: ReturnType<typeof setInterval> | undefined;
  onCleanup(() => clearInterval(timer));

  const errors = createMemo(() => validate(p(), source() !== null, hasMask()));
  const model = () => IMAGE_MODELS.find((m) => m.id === p().modelId)!;
  const speed = () => (new URLSearchParams(location.search).getAll("mock").includes("fast-images") ? 20 : 1);

  const pickModel = (id: string) => {
    const m = IMAGE_MODELS.find((x) => x.id === id);
    if (m) set({ modelId: id, steps: m.defaultSteps, guidance: m.defaultGuidance });
  };

  const generate = () => {
    setTried(true);
    if (Object.keys(errors()).length || progress() !== null) return;
    const params = p();
    const seed = params.seed ?? randomSeed();
    const { w, h } = sizeFor(params);
    const steps = params.mode === "upscale" ? 10 : params.steps;
    const total = Math.min(6000, estimateSeconds(params) * 120) / speed();
    const item: GalleryItem = { id: newId(), prompt: params.prompt.trim() || (params.mode === "upscale" ? "Upscaled image" : "Variation"), negative: params.negative, modelId: params.modelId, mode: params.mode, seed, w, h, steps, guidance: params.guidance, favourite: false, at: Date.now() };
    setCurrent(item);
    setProgress(0);
    const started = performance.now();
    timer = setInterval(() => {
      const f = Math.min(1, (performance.now() - started) / total);
      setProgress(f);
      if (f >= 1) {
        clearInterval(timer);
        setProgress(null);
        const next = [item, ...gallery()];
        setGallery(next);
        saveGallery(next);
        notify("Image ready", `${w}×${h}, seed ${seed}. Saved to the gallery on this computer.`, "success");
      }
    }, 50);
  };
  const cancel = () => {
    clearInterval(timer);
    setProgress(null);
    setCurrent(null);
  };

  const onSource = (f: File | undefined) => {
    if (!f || !/^image\/(png|jpeg|webp)$/.test(f.type) || f.size > 20 * 1024 * 1024) {
      notify("Not used", "Choose a PNG, JPEG or WebP image under 20 MB.", "warning");
      return;
    }
    const r = new FileReader();
    r.onload = () => typeof r.result === "string" && (setSource(r.result), setHasMask(false));
    r.readAsDataURL(f);
  };

  const toggleFav = (id: string) => {
    const next = gallery().map((g) => (g.id === id ? { ...g, favourite: !g.favourite } : g));
    setGallery(next);
    saveGallery(next);
  };
  const reuse = (g: GalleryItem) => {
    set({ prompt: g.prompt, negative: g.negative, modelId: g.modelId, seed: g.seed, steps: g.steps, guidance: g.guidance, mode: g.mode === "upscale" || g.mode === "variations" ? "txt2img" : g.mode });
    notify("Prompt reused", "Same prompt, model, seed and settings.");
  };
  const remove = (id: string) => {
    const next = gallery().filter((g) => g.id !== id);
    setGallery(next);
    saveGallery(next);
  };
  const shown = () => (filter() === "favourites" ? gallery().filter((g) => g.favourite) : gallery());

  return (
    <div class="screen images">
      <ScreenHeader route="images" lede="Make and edit images with local models. Built in: no plugin, no upload, nothing leaves this computer." actions={<IllustrativeFlag />} />
      <div class="img-layout">
        <form
          class="img-controls card"
          onSubmit={(e) => {
            e.preventDefault();
            generate();
          }}
        >
          <div class="img-modes" role="tablist" aria-label="Mode">
            <For each={MODES}>
              {(m) => (
                <button type="button" role="tab" aria-selected={p().mode === m} class="img-mode" onClick={() => set({ mode: m })}>
                  {MODE_LABELS[m]}
                </button>
              )}
            </For>
          </div>

          <label class="img-field">
            <span>Model</span>
            <select value={p().modelId} onChange={(e) => pickModel(e.currentTarget.value)}>
              <For each={IMAGE_MODELS}>
                {(m) => (
                  <option value={m.id}>
                    {m.name} · {m.format} {m.quant} · {m.sizeGiB} GiB
                  </option>
                )}
              </For>
            </select>
          </label>

          <Show when={p().mode !== "txt2img"}>
            <div class="img-field">
              <span>Source image</span>
              <Show when={source()} fallback={<SourceDrop onFile={onSource} />}>
                {(src) => (
                  <div class="img-source">
                    <Show when={p().mode === "inpaint"} fallback={<img src={src()} alt="Source image" />}>
                      <MaskCanvas src={src()} onPainted={() => setHasMask(true)} onCleared={() => setHasMask(false)} />
                    </Show>
                    <button type="button" class="btn btn--tertiary btn--sm" onClick={() => setSource(null)}>
                      Remove
                    </button>
                  </div>
                )}
              </Show>
            </div>
          </Show>

          <Show when={p().mode !== "upscale"}>
            <label class="img-field">
              <span>Prompt</span>
              <textarea rows="3" maxlength="2000" value={p().prompt} placeholder="A lighthouse on a cliff at dusk, oil painting, warm light" onInput={(e) => set({ prompt: e.currentTarget.value })} aria-invalid={tried() && !!errors().prompt} aria-describedby="err-prompt" />
            </label>
            <Show when={tried() && errors().prompt}>
              <span class="img-error" id="err-prompt">
                {errors().prompt}
              </span>
            </Show>
            <label class="img-field">
              <span>Negative prompt</span>
              <input maxlength="1000" value={p().negative} placeholder="blurry, text, watermark" onInput={(e) => set({ negative: e.currentTarget.value })} />
            </label>
            <fieldset class="img-aspects">
              <legend>Aspect ratio</legend>
              <For each={ASPECTS}>
                {(a) => (
                  <label class="img-chip" data-on={p().aspect === a.id}>
                    <input type="radio" name="aspect" value={a.id} checked={p().aspect === a.id} onChange={() => set({ aspect: a.id })} />
                    {a.id}
                  </label>
                )}
              </For>
            </fieldset>
          </Show>

          <Show when={p().mode === "upscale"}>
            <fieldset class="img-aspects">
              <legend>Scale</legend>
              <For each={[2, 4] as const}>
                {(s) => (
                  <label class="img-chip" data-on={p().scale === s}>
                    <input type="radio" name="scale" checked={p().scale === s} onChange={() => set({ scale: s })} />
                    {s}×
                  </label>
                )}
              </For>
            </fieldset>
          </Show>

          <label class="img-toggle">
            <input type="checkbox" checked={advanced()} onChange={(e) => setAdvanced(e.currentTarget.checked)} />
            <span>Advanced settings</span>
          </label>
          <Show when={advanced()}>
            <div class="img-advanced">
              <label class="img-field">
                <span>Seed</span>
                <input inputmode="numeric" placeholder="Random" value={p().seed ?? ""} onChange={(e) => set({ seed: e.currentTarget.value.trim() === "" ? null : Number(e.currentTarget.value) })} aria-invalid={!!errors().seed} />
              </label>
              <label class="img-field">
                <span>Steps</span>
                <input type="number" min="1" max="150" value={p().steps} onChange={(e) => set({ steps: Number(e.currentTarget.value) })} aria-invalid={!!errors().steps} />
              </label>
              <label class="img-field">
                <span>Guidance</span>
                <input type="number" min="0" max="30" step="0.5" value={p().guidance} onChange={(e) => set({ guidance: Number(e.currentTarget.value) })} aria-invalid={!!errors().guidance} />
              </label>
              <label class="img-field">
                <span>Sampler</span>
                <select value={p().sampler} onChange={(e) => set({ sampler: e.currentTarget.value as GenParams["sampler"] })}>
                  <For each={SAMPLERS}>{(s) => <option value={s}>{s}</option>}</For>
                </select>
              </label>
              <Show when={p().mode === "img2img" || p().mode === "inpaint" || p().mode === "variations"}>
                <label class="img-field">
                  <span>Strength {p().strength.toFixed(2)}</span>
                  <input type="range" min="0" max="1" step="0.05" value={p().strength} onInput={(e) => set({ strength: Number(e.currentTarget.value) })} />
                </label>
              </Show>
              <For each={(["seed", "steps", "guidance"] as const).filter((k) => errors()[k])}>
                {(k) => <span class="img-error">{errors()[k]}</span>}
              </For>
            </div>
          </Show>

          <Show when={tried() && errors().mode}>
            <span class="img-error">{errors().mode}</span>
          </Show>
          <div class="img-go">
            <Show
              when={progress() === null}
              fallback={
                <button type="button" class="btn btn--secondary" onClick={cancel} data-press>
                  <Icon name="stop" size={16} /> Cancel
                </button>
              }
            >
              <button type="submit" class="btn btn--primary" data-press>
                <Icon name="images" size={16} /> Generate
              </button>
            </Show>
            <span class="hint">
              About {estimateSeconds(p())} s on {model().name} here
            </span>
          </div>
        </form>

        <section class="img-stage card" aria-label="Result">
          <Show when={current()} fallback={<p class="img-stage__empty">Your image appears here, refining step by step as it is made.</p>}>
            {(item) => (
              <figure class="img-result">
                <Art item={item()} progress={progress() ?? 1} />
                <figcaption>
                  <Show when={progress() !== null} fallback={<span>{item().prompt}</span>}>
                    <span role="status" aria-live="polite">
                      Step {Math.max(1, Math.round((progress() ?? 0) * item().steps))} of {item().steps}
                    </span>
                  </Show>
                  <span class="mono hint">
                    {item().w}×{item().h} · seed {item().seed}
                  </span>
                </figcaption>
                <Show when={progress() !== null}>
                  <div class="img-progress" role="progressbar" aria-label="Generating" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round((progress() ?? 0) * 100)}>
                    <span style={{ transform: `scaleX(${progress() ?? 0})` }} />
                  </div>
                </Show>
              </figure>
            )}
          </Show>
        </section>
      </div>

      <section class="img-gallery" aria-labelledby="gallery-title">
        <div class="img-gallery__head">
          <h2 id="gallery-title">Gallery</h2>
          <fieldset class="segmented segmented--sm">
            <legend class="sr-only">Show</legend>
            <label class="segmented__option">
              <input type="radio" name="gallery-filter" checked={filter() === "all"} onChange={() => setFilter("all")} />
              <span>All ({gallery().length})</span>
            </label>
            <label class="segmented__option">
              <input type="radio" name="gallery-filter" checked={filter() === "favourites"} onChange={() => setFilter("favourites")} />
              <span>Favourites</span>
            </label>
          </fieldset>
        </div>
        <ul class="img-grid" role="list">
          <For each={shown()} fallback={<li class="hint">{gallery().length ? "No favourites yet." : "Images you make are kept here, on this computer."}</li>}>
            {(g) => (
              <li class="img-tile" data-fav={g.favourite}>
                <Art item={g} progress={1} small />
                <div class="img-tile__bar">
                  <span class="img-tile__prompt">{g.prompt}</span>
                  <button type="button" class="img-icon" aria-label={g.favourite ? `Unfavourite ${g.prompt}` : `Favourite ${g.prompt}`} aria-pressed={g.favourite} onClick={() => toggleFav(g.id)}>
                    {g.favourite ? "★" : "☆"}
                  </button>
                  <button type="button" class="img-icon" aria-label={`Reuse prompt: ${g.prompt}`} onClick={() => reuse(g)}>
                    <Icon name="refresh" size={14} />
                  </button>
                  <a class="img-icon" aria-label={`Export ${g.prompt}`} download={`cerebro-${g.seed}.svg`} href={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(artSvg(g))}`}>
                    <Icon name="download" size={14} />
                  </a>
                  <button type="button" class="img-icon" aria-label={`Delete ${g.prompt}`} onClick={() => remove(g.id)}>
                    <Icon name="trash" size={14} />
                  </button>
                </div>
              </li>
            )}
          </For>
        </ul>
      </section>
    </div>
  );
}

function SourceDrop(props: { onFile: (f: File | undefined) => void }): JSX.Element {
  const [over, setOver] = createSignal(false);
  return (
    <label
      class="img-drop"
      data-over={over()}
      onDragOver={(e) => (e.preventDefault(), setOver(true))}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        props.onFile(e.dataTransfer?.files[0]);
      }}
    >
      <Icon name="upload" size={20} />
      <span>Drop an image or choose one</span>
      <input class="sr-only" type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => props.onFile(e.currentTarget.files?.[0])} />
    </label>
  );
}

/** Paint the area to change. The mask stays in this screen; only its shape would go to the model. */
function MaskCanvas(props: { src: string; onPainted: () => void; onCleared: () => void }): JSX.Element {
  let canvas!: HTMLCanvasElement;
  let drawing = false;
  const [brush, setBrush] = createSignal(28);
  const paint = (e: PointerEvent) => {
    if (!drawing) return;
    const r = canvas.getBoundingClientRect();
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "rgba(80, 160, 255, 0.55)";
    ctx.beginPath();
    ctx.arc(((e.clientX - r.left) / r.width) * canvas.width, ((e.clientY - r.top) / r.height) * canvas.height, brush(), 0, Math.PI * 2);
    ctx.fill();
    props.onPainted();
  };
  const clear = () => {
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    props.onCleared();
  };
  return (
    <div class="img-mask">
      <div class="img-mask__stack">
        <img src={props.src} alt="Source image" />
        <canvas
          ref={canvas}
          width="512"
          height="512"
          aria-label="Mask: paint over the area to change"
          role="img"
          onPointerDown={(e) => {
            drawing = true;
            canvas.setPointerCapture(e.pointerId);
            paint(e);
          }}
          onPointerMove={paint}
          onPointerUp={() => (drawing = false)}
        />
      </div>
      <div class="img-mask__tools">
        <label>
          Brush <input type="range" min="6" max="80" value={brush()} onInput={(e) => setBrush(Number(e.currentTarget.value))} />
        </label>
        <button type="button" class="btn btn--tertiary btn--sm" onClick={clear}>
          Clear mask
        </button>
      </div>
    </div>
  );
}

interface Shape {
  kind: "circle" | "rect";
  x: number;
  y: number;
  r: number;
  h: number;
  l: number;
  rot: number;
}

function shapesFor(item: GalleryItem): { hue: number; shapes: Shape[] } {
  const rand = rng(item.seed ^ hashPrompt(item.prompt));
  const hue = Math.round(rand() * 360);
  const shapes = Array.from({ length: 9 }, (): Shape => ({ kind: rand() > 0.35 ? "circle" : "rect", x: rand() * 100, y: rand() * 100, r: 12 + rand() * 38, h: (hue + Math.round(rand() * 120) - 60 + 360) % 360, l: 45 + rand() * 35, rot: rand() * 90 }));
  return { hue, shapes };
}

/** The illustrative artwork; while generating, noise fades and detail sharpens with progress. */
function Art(props: { item: GalleryItem; progress: number; small?: boolean }): JSX.Element {
  const data = createMemo(() => shapesFor(props.item));
  const fid = `noise-${props.item.id}${props.small ? "-s" : ""}`;
  return (
    <svg class="img-art" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice" role="img" aria-label={`Image: ${props.item.prompt}`} style={{ "aspect-ratio": `${props.item.w} / ${props.item.h}` }}>
      <defs>
        <filter id={fid}>
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed={props.item.seed % 1000} />
          <feColorMatrix type="saturate" values="0" />
        </filter>
      </defs>
      <rect width="100" height="100" fill={`oklch(24% 0.06 ${data().hue})`} />
      <g style={{ filter: `blur(${(1 - props.progress) * 6}px)` }}>
        <For each={data().shapes}>
          {(s) =>
            s.kind === "circle" ? (
              <circle cx={s.x} cy={s.y} r={s.r} fill={`oklch(${s.l}% 0.15 ${s.h} / 0.7)`} />
            ) : (
              <rect x={s.x - s.r / 2} y={s.y - s.r / 2} width={s.r} height={s.r} fill={`oklch(${s.l}% 0.13 ${s.h} / 0.6)`} transform={`rotate(${s.rot} ${s.x} ${s.y})`} />
            )
          }
        </For>
      </g>
      <rect width="100" height="100" filter={`url(#${fid})`} opacity={Math.max(0, 0.9 - props.progress * 0.9)} />
    </svg>
  );
}

function artSvg(item: GalleryItem): string {
  const { hue, shapes } = shapesFor(item);
  const body = shapes
    .map((s) => (s.kind === "circle" ? `<circle cx="${s.x.toFixed(2)}" cy="${s.y.toFixed(2)}" r="${s.r.toFixed(2)}" fill="oklch(${s.l.toFixed(1)}% 0.15 ${s.h} / 0.7)"/>` : `<rect x="${(s.x - s.r / 2).toFixed(2)}" y="${(s.y - s.r / 2).toFixed(2)}" width="${s.r.toFixed(2)}" height="${s.r.toFixed(2)}" fill="oklch(${s.l.toFixed(1)}% 0.13 ${s.h} / 0.6)" transform="rotate(${s.rot.toFixed(1)} ${s.x.toFixed(2)} ${s.y.toFixed(2)})"/>`))
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="${item.w}" height="${item.h}" preserveAspectRatio="xMidYMid slice"><rect width="100" height="100" fill="oklch(24% 0.06 ${hue})"/>${body}</svg>`;
}
