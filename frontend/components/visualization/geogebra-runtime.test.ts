// @ts-nocheck -- Bun provides the test runtime types without adding a browser bundle dependency.
import { describe, expect, test } from "bun:test";

import {
  applyGeoGebraRenderHints,
  configureGeoGebraView,
  executeGeoGebraCommands,
} from "./geogebra-runtime";


describe("executeGeoGebraCommands", () => {
  test("executes commands sequentially and reports success", () => {
    const calls: string[] = [];
    const api = {
      evalCommand(command: string) {
        calls.push(command);
        return true;
      },
      getXML: () => "<xml />",
    };

    const result = executeGeoGebraCommands(api, ["A = (0, 0)", "B = (1, 0)"]);

    expect(result).toEqual({ success: true, executed: 2 });
    expect(calls).toEqual(["A = (0, 0)", "B = (1, 0)"]);
  });

  test("stops on false and restores the snapshot", () => {
    const calls: string[] = [];
    const restored: string[] = [];
    const api = {
      evalCommand(command: string) {
        calls.push(command);
        return command !== "bad()";
      },
      getXML: () => "before",
      setXML: (xml: string) => restored.push(xml),
    };

    const result = executeGeoGebraCommands(api, ["ok()", "bad()", "never()"]);

    expect(result.success).toBeFalse();
    expect(result.executed).toBe(1);
    expect(result.failure).toMatchObject({
      index: 1,
      command: "bad()",
      reason: "returned_false",
      rollback: "restored",
    });
    expect(calls).toEqual(["ok()", "bad()"]);
    expect(restored).toEqual(["before"]);
  });

  test("reports exceptions and clears when snapshot restore is unavailable", () => {
    let cleared = 0;
    const api = {
      evalCommand() {
        throw new Error("runtime failure");
      },
      newConstruction: () => {
        cleared += 1;
      },
    };

    const result = executeGeoGebraCommands(api, ["bad()"]);

    expect(result.failure).toMatchObject({
      index: 0,
      reason: "exception",
      detail: "runtime failure",
      rollback: "cleared",
    });
    expect(cleared).toBe(1);
  });
});


describe("structured rendering controls", () => {
  test("selects documented perspectives and enables special environments", () => {
    const calls: unknown[][] = [];
    const api = {
      evalCommand: () => true,
      enableCAS: (value: boolean) => calls.push(["cas", value]),
      enable3D: (value: boolean) => calls.push(["3d", value]),
      setPerspective: (value: string) => calls.push(["perspective", value]),
    };

    configureGeoGebraView(api, "cas");

    expect(calls).toEqual([
      ["cas", true],
      ["3d", false],
      ["perspective", "4"],
    ]);
  });

  test("applies safe styling, viewport, and interaction API calls", () => {
    const calls: unknown[][] = [];
    const api = new Proxy(
      { evalCommand: () => true },
      {
        get(target, property) {
          if (property in target) return target[property];
          return (...args: unknown[]) => calls.push([property, ...args]);
        },
      },
    );

    applyGeoGebraRenderHints(api, "geometry_2d", {
      styles: [
        {
          label: "A",
          color: "#FF0080",
          line_thickness: 3,
          point_size: 5,
          label_visible: true,
          fixed: false,
        },
      ],
      viewport: {
        x_min: -5,
        x_max: 5,
        y_min: -4,
        y_max: 4,
        axes_visible: false,
        grid_visible: true,
      },
      interaction: {
        movable_points: ["A"],
        animated_objects: ["slider1"],
        animation: "start",
      },
    });

    expect(calls).toContainEqual(["setColor", "A", 255, 0, 128]);
    expect(calls).toContainEqual(["setCoordSystem", -5, 5, -4, 4]);
    expect(calls).toContainEqual(["setAxesVisible", false, false]);
    expect(calls).toContainEqual(["setGridVisible", true]);
    expect(calls).toContainEqual(["setFixed", "A", false, true]);
    expect(calls).toContainEqual(["setAnimating", "slider1", true]);
    expect(calls).toContainEqual(["startAnimation"]);
  });
});
