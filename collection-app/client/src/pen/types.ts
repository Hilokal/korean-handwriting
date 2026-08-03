// Copied verbatim from src/dataCollection/types.ts — this is the wire/storage
// shape the training pipeline (handwriting_dataset.py) consumes.

export interface PageInfo {
  section: number;
  owner: number;
  book: number;
  page: number;
}

export interface Angle {
  tx: number;
  ty: number;
  twist: number;
}

export interface RecordedDot {
  x: number;
  y: number;
  f: number;
  dotType: number; // 0=DOWN, 1=MOVE, 2=UP, 3=HOVER
  timeStamp: number;
  pageInfo: PageInfo;
  angle?: Angle;
  timeDiff?: number;
  isPlate?: boolean;
  penTipType: number;
}

export interface RecordingSegment {
  startTime: number;
  endTime: number;
  dotCount: number;
  dots: RecordedDot[];
  metadata: {
    exportedAt: string;
    penMac?: string;
  };
}
