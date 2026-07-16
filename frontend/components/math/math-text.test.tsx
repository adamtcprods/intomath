// @ts-nocheck -- Bun provides test globals without adding them to the browser bundle.
import { describe, expect, test } from "bun:test";
import katex from "katex";
import { renderToStaticMarkup } from "react-dom/server";

import { MathText } from "./math-text";


describe("MathText", () => {
  test("passes a non-empty gcd formula through react-katex", () => {
    const formula = String.raw`\gcd(a^n+b,b^n+a)=g`;

    expect(() =>
      katex.renderToString(formula, { throwOnError: true }),
    ).not.toThrow();

    const markup = renderToStaticMarkup(
      <MathText
        latex={[formula]}
        text="The gcd expression is eventually constant."
      />,
    );

    expect(markup).toContain("katex-display");
    expect(markup).toContain("gcd");
  });
});
