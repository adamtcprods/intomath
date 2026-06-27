import { Fragment, type ReactNode } from "react";
import { BlockMath, InlineMath } from "react-katex";

import { cn } from "@/lib/utils";

interface MathTextProps {
  text: string;
  latex?: string[] | string | null;
  className?: string;
}

export function MathText({ text, latex, className }: MathTextProps) {
  const formulas = normalizeLatexList(latex);

  return (
    <div className={cn("space-y-3", className)}>
      <p className="math-copy text-sm leading-7 text-foreground/90">
        {renderInlineMath(text)}
      </p>
      {formulas.length > 0 ? (
        <div className="space-y-3 rounded-xl bg-secondary/70 p-4">
          {formulas.map((formula, index) => (
            <BlockMath key={`${formula}-${index}`}>{formula}</BlockMath>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function normalizeLatexList(latex: MathTextProps["latex"]): string[] {
  const formulas = Array.isArray(latex) ? latex : latex ? [latex] : [];
  return formulas.map(normalizeLatexFormula).filter(Boolean);
}

function normalizeLatexFormula(value: string): string {
  let formula = value
    .trim()
    .replace(/^```(?:latex|tex|math)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();

  const delimiterPairs: Array<[string, string]> = [
    ["$$", "$$"],
    ["\\[", "\\]"],
    ["\\(", "\\)"],
    ["$", "$"],
  ];

  let changed = true;
  while (changed) {
    changed = false;
    for (const [open, close] of delimiterPairs) {
      if (formula.startsWith(open) && formula.endsWith(close)) {
        formula = formula
          .slice(open.length, formula.length - close.length)
          .trim();
        changed = true;
      }
    }
  }

  return formula;
}

function renderInlineMath(text: string): ReactNode[] {
  const segments: ReactNode[] = [];
  const inlineMathPattern = /\\\((.+?)\\\)|\$([^$\n]+?)\$/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = inlineMathPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push(text.slice(lastIndex, match.index));
    }

    const formula = normalizeLatexFormula(match[1] ?? match[2] ?? "");
    segments.push(
      <InlineMath key={`math-${match.index}-${formula}`}>{formula}</InlineMath>,
    );
    lastIndex = inlineMathPattern.lastIndex;
  }

  if (lastIndex < text.length) {
    segments.push(text.slice(lastIndex));
  }

  return segments.map((segment, index) => (
    <Fragment key={typeof segment === "string" ? `${index}-${segment}` : index}>
      {segment}
    </Fragment>
  ));
}
