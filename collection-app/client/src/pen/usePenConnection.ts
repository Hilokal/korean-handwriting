import { useEffect, useRef, useState } from "react";
import { PenHelper, PenMessageType, PenController } from "web_pen_sdk";
import type { Dot } from "web_pen_sdk/dist/Util/type";
import { RecordedDot } from "./types";

/**
 * Wires the web_pen_sdk global callbacks into React state. Extracted from the
 * original DataCollectionView: callbacks are singletons on PenHelper, so the
 * dot handler goes through a ref to avoid stale closures.
 */
export function usePenConnection(onDot: (dot: RecordedDot) => void) {
  const [controller, setController] = useState<PenController>();
  const [isConnected, setIsConnected] = useState(false);
  const onDotRef = useRef(onDot);
  onDotRef.current = onDot;

  useEffect(() => {
    PenHelper.messageCallback = (mac: string, type: number, _args: unknown) => {
      switch (type) {
        case PenMessageType.PEN_SETTING_INFO: {
          const _controller = PenHelper.pens.filter(
            (c: PenController) => c.info.MacAddress === mac,
          )[0];
          setController(_controller);
          setIsConnected(true);
          break;
        }
        case PenMessageType.PEN_DISCONNECTED:
          setController(undefined);
          setIsConnected(false);
          break;
      }
    };

    PenHelper.dotCallback = (_mac: string, dot: Dot) => {
      onDotRef.current({
        x: dot.x,
        y: dot.y,
        f: dot.f,
        dotType: dot.dotType,
        timeStamp: dot.timeStamp,
        penTipType: dot.penTipType,
        timeDiff: (dot as any).timeDiff,
        isPlate: (dot as any).isPlate,
        pageInfo: {
          section: dot.pageInfo.section,
          owner: dot.pageInfo.owner,
          book: dot.pageInfo.book,
          page: dot.pageInfo.page,
        },
        angle: dot.angle
          ? {
              tx: (dot.angle as any).tx,
              ty: (dot.angle as any).ty,
              twist: (dot.angle as any).twist,
            }
          : undefined,
      });
    };
  }, []);

  return {
    controller,
    isConnected,
    penMac: controller?.info?.MacAddress as string | undefined,
  };
}
