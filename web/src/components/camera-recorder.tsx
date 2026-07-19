"use client";

import { Camera, CameraOff, Clock3, RefreshCw, RotateCcw, Square, Video } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

const MAX_RECORD_SECONDS = 30;

function supportedMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  return ["video/webm;codecs=vp8", "video/webm", "video/mp4"].find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

interface CameraRecorderProps {
  disabled?: boolean;
  onCapture: (file: File | null, previewUrl?: string) => void;
  onError: (message: string) => void;
}

export function CameraRecorder({ disabled, onCapture, onError }: CameraRecorderProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);
  const onCaptureRef = useRef(onCapture);
  const onErrorRef = useRef(onError);
  const [cameraReady, setCameraReady] = useState(false);
  const [requesting, setRequesting] = useState(true);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const secondsRemaining = Math.max(0, Math.ceil(MAX_RECORD_SECONDS - elapsed));

  useEffect(() => {
    onCaptureRef.current = onCapture;
    onErrorRef.current = onError;
  }, [onCapture, onError]);

  useEffect(() => {
    const video = videoRef.current;
    if (!recordedUrl || !video) return;
    // React can reuse the live <video> node for playback. A MediaStream in
    // srcObject takes precedence over src, so detach it before loading the Blob.
    video.pause();
    video.srcObject = null;
    video.src = recordedUrl;
    video.load();
  }, [recordedUrl]);

  const stopTimer = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
  }, []);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraReady(false);
  }, []);

  const startCamera = useCallback(async (facing: "environment" | "user") => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setRequesting(false);
      onErrorRef.current("Camera recording is not supported here. Use Upload instead.");
      return;
    }
    setRequesting(true);
    stopStream();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.src = "";
        videoRef.current.muted = true;
        await videoRef.current.play().catch(() => undefined);
      }
      setCameraReady(true);
    } catch {
      onErrorRef.current("Camera access was blocked. Check browser permissions or use Upload.");
    } finally {
      setRequesting(false);
    }
  }, [stopStream]);

  useEffect(() => {
    // Camera access is the external system this effect synchronizes with.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    startCamera(facingMode);
    return () => {
      stopTimer();
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      stopStream();
    };
  }, [facingMode, startCamera, stopStream, stopTimer]);

  useEffect(() => {
    // React Strict Mode runs a setup → cleanup → setup cycle in development.
    // Restore the flag in setup so a valid recording is not discarded after
    // that development-only remount.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const stopRecording = useCallback(() => {
    stopTimer();
    const recorder = recorderRef.current;
    if (recorder?.state === "recording") recorder.stop();
    setRecording(false);
  }, [stopTimer]);

  function startRecording() {
    const stream = streamRef.current;
    if (!stream || disabled) return;
    const mimeType = supportedMimeType();
    const options: MediaRecorderOptions = { videoBitsPerSecond: 2_500_000 };
    if (mimeType) options.mimeType = mimeType;
    try {
      const recorder = new MediaRecorder(stream, options);
      chunksRef.current = [];
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data);
      };
      recorder.onerror = () => onErrorRef.current("Recording was interrupted. Please try again.");
      recorder.onstop = () => {
        const type = chunksRef.current[0]?.type || mimeType || "video/webm";
        const blob = new Blob(chunksRef.current, { type });
        if (!blob.size) {
          onErrorRef.current("Nothing was recorded. Please try again.");
          return;
        }
        if (!mountedRef.current) return;
        if (videoRef.current) {
          videoRef.current.pause();
          videoRef.current.srcObject = null;
        }
        if (recordedUrl) URL.revokeObjectURL(recordedUrl);
        const url = URL.createObjectURL(blob);
        const extension = type.includes("mp4") ? "mp4" : "webm";
        const file = new File([blob], `trainer-recording.${extension}`, { type });
        setRecordedUrl(url);
        stopStream();
        onCaptureRef.current(file, url);
      };
      recorder.start(250);
      setElapsed(0);
      setRecording(true);
      const startedAt = Date.now();
      timerRef.current = setInterval(() => {
        const seconds = Math.min(MAX_RECORD_SECONDS, (Date.now() - startedAt) / 1000);
        setElapsed(seconds);
        if (seconds >= MAX_RECORD_SECONDS) stopRecording();
      }, 100);
    } catch {
      onErrorRef.current("This browser could not start video recording. Use Upload instead.");
    }
  }

  async function retake() {
    if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    setRecordedUrl(null);
    setElapsed(0);
    onCaptureRef.current(null);
    await startCamera(facingMode);
  }

  function switchCamera() {
    if (recording || recordedUrl) return;
    setFacingMode((current) => (current === "environment" ? "user" : "environment"));
  }

  return (
    <div className="camera-shell">
      <div className="camera-frame">
        {recordedUrl ? (
          <video ref={videoRef} className="media-preview" src={recordedUrl} controls playsInline preload="metadata" />
        ) : (
          <video ref={videoRef} className="media-preview live-camera" autoPlay muted playsInline />
        )}
        {requesting && !recordedUrl && (
          <div className="camera-overlay"><RefreshCw className="spin" size={20} /> Opening camera…</div>
        )}
        {!requesting && !cameraReady && !recordedUrl && (
          <div className="camera-overlay"><CameraOff size={22} /> Camera unavailable</div>
        )}
        {cameraReady && !recording && !recordedUrl && (
          <div className="record-limit"><Clock3 size={15} /><span><strong>{MAX_RECORD_SECONDS} seconds max</strong>Stops automatically</span></div>
        )}
        {recording && (
          <>
            <div className={`record-badge${secondsRemaining <= 5 ? " ending" : ""}`}><span /> REC&nbsp; {secondsRemaining}s left</div>
            <div className={`record-limit-progress${secondsRemaining <= 5 ? " ending" : ""}`} aria-hidden="true"><i style={{ width: `${Math.min(100, (elapsed / MAX_RECORD_SECONDS) * 100)}%` }} /></div>
          </>
        )}
      </div>

      <div className="camera-actions">
        {recordedUrl ? (
          <button className="secondary-button" type="button" onClick={retake} disabled={disabled}>
            <RotateCcw size={17} /> Retake
          </button>
        ) : (
          <>
            <button
              className={recording ? "stop-button" : "record-button"}
              type="button"
              onClick={recording ? stopRecording : startRecording}
              disabled={!cameraReady || disabled}
            >
              {recording ? <Square size={16} fill="currentColor" /> : <Video size={18} />}
              {recording ? "Stop recording" : "Start recording"}
            </button>
            <button className="icon-button" type="button" onClick={switchCamera} disabled={!cameraReady || recording || disabled} aria-label="Switch camera">
              <Camera size={18} />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
