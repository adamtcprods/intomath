declare module "react-katex" {
  import type { ReactElement, ReactNode } from "react";

  interface BaseMathProps {
    children?: ReactNode;
    math?: string;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
  }

  export function BlockMath(props: BaseMathProps): ReactElement;
  export function InlineMath(props: BaseMathProps): ReactElement;
}
