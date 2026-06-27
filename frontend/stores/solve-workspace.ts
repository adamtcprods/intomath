import { create } from "zustand";

interface SolveWorkspaceState {
  input: string;
  imageBase64: string | null;
  imageMimeType: string | null;
  setInput: (input: string) => void;
  setImageBase64: (imageBase64: string | null) => void;
  setImageMimeType: (imageMimeType: string | null) => void;
}

export const useSolveWorkspaceStore = create<SolveWorkspaceState>((set) => ({
  input: "",
  imageBase64: null,
  imageMimeType: null,
  setInput: (input) => set({ input }),
  setImageBase64: (imageBase64) => set({ imageBase64 }),
  setImageMimeType: (imageMimeType) => set({ imageMimeType }),
}));
