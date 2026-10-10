import type { CatalogModel } from "./types";

/**
 * Models setup may recommend before anything is downloaded. Empty until the Model Hub's live
 * Hugging Face search is wired in: setup then recommends from what the Hub finds and from the
 * library, never from a fixed list of example figures.
 */
export const CURRENT_CATALOG: readonly CatalogModel[] = [];
