import { PenHelper, PenController } from "web_pen_sdk";
import { useI18n } from "../i18n";

interface ConnectionButtonProps {
  isConnected: boolean;
  controller: PenController | undefined;
}

export function ConnectionButton({ isConnected, controller }: ConnectionButtonProps) {
  const { s } = useI18n();
  const handleClick = () => {
    if (isConnected) {
      if (controller) PenHelper.disconnect(controller);
    } else {
      PenHelper.scanPen();
    }
  };

  return (
    <button className={isConnected ? "btn btn-secondary" : "btn btn-primary"} onClick={handleClick}>
      {isConnected ? s.disconnectPen : s.connectPen}
    </button>
  );
}
