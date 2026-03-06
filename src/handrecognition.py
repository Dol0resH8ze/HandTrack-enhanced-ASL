import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import os

# config
MODEL_FILE     = "asl_model.pkl"
LABELS_FILE    = "asl_labels.pkl"
HOLD_SECONDS   = 1.0
MIN_CONFIDENCE = 0.6

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODE_COUNTING = "COUNTING"
MODE_ASL      = "ASL"

import features as F


def load_model():
    try:
        with open(MODEL_FILE,  "rb") as f: model  = pickle.load(f)
        with open(LABELS_FILE, "rb") as f: labels = pickle.load(f)
        return model, labels
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 1_collect_data.py then 2_train_model.py first.")
        exit(1)


def predict_letter(model, labels, hand_landmarks):
    feat  = F.from_mediapipe(hand_landmarks)
    proba = model.predict_proba(feat)[0]
    idx   = np.argmax(proba)
    return labels[idx], proba[idx]


def count_fingers(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    count = 0
    if handedness_label == "Right":
        if lm[4].x < lm[3].x: count += 1
    else:
        if lm[4].x > lm[3].x: count += 1
    for tip, pip in zip(finger_tips, finger_pips):
        if lm[tip].y < lm[pip].y: count += 1
    return count


def main():
    model, labels = load_model()
    print(f"Model loaded. Recognizes: {labels}")

    cap  = cv2.VideoCapture(1)
    mode = MODE_COUNTING

    # ASL state
    phrase         = []    # list of letters and spaces e.g. ['H','E','L','L','O',' ','W']
    hold_start     = None
    last_letter    = None
    pending_letter = None
    pending_conf   = 0.0
    show_final     = False
    final_phrase   = ""

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result  = hands.process(rgb)
            h, w, _ = frame.shape
            now     = time.time()

            # Read key first so ASL mode can use it
            key = cv2.waitKey(1) & 0xFF

            # Draw landmarks
            if result.multi_hand_landmarks:
                for hl in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            # mode bar
            cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(frame, f" {mode}   [TAB] toggle / [SPACE] space / [ENTER] done / [C] clear",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

           # counting mode
            if mode == MODE_COUNTING:
                total      = 0
                hand_counts = []

                if result.multi_hand_landmarks and result.multi_handedness:
                    num_hands = len(result.multi_hand_landmarks)
                    for idx, hl in enumerate(result.multi_hand_landmarks):
                        label = result.multi_handedness[idx].classification[0].label
                        cnt   = count_fingers(hl, label)
                        hand_counts.append((label, cnt, hl))
                        total += cnt

                    for label, cnt, hl in hand_counts:
                        
                        for tip_id in [4, 8, 12, 16, 20]:
                            cx = int(hl.landmark[tip_id].x * w)
                            cy = int(hl.landmark[tip_id].y * h)
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                        
                        cx = int(hl.landmark[0].x * w)
                        cy = int(hl.landmark[0].y * h)
                        if cnt > 0:
                            cv2.putText(frame, str(cnt),
                                        (cx - 90, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                          

               #sum 
                    if len(hand_counts) == 2:
                        cv2.putText(frame, f"Sum: {total}",
                                    (w // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

                

          # asl mode
            else:
                # key actions
                if key == ord(' ') and not show_final:
                    # space between words
                    if phrase and phrase[-1] != ' ':
                        phrase.append(' ')
                    hold_start     = None
                    last_letter    = None
                    pending_letter = None

                elif key == 13 and phrase:  # enter to finalizeeeeee
                    final_phrase = "".join(phrase).strip()
                    show_final   = True
                    phrase       = []
                    hold_start   = None
                    pending_letter = None

                    # save to txt and open it
                    txt_path = "asl_phrase.txt"
                    with open(txt_path, "w") as f:
                        f.write(final_phrase)
                    os.startfile(txt_path)
                    # close camera and window
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                #  letter detection 
                if not show_final and result.multi_hand_landmarks:
                    hl = result.multi_hand_landmarks[0]
                    letter, conf = predict_letter(model, labels, hl)

                    if conf >= MIN_CONFIDENCE:
                        if letter != last_letter:
                            hold_start     = now
                            last_letter    = letter
                            pending_letter = letter
                            pending_conf   = conf
                        else:
                            pending_conf = conf
                    else:
                        hold_start     = None
                        last_letter    = None
                        pending_letter = None

                    # confirm letter after hold
                    if hold_start and pending_letter and (now - hold_start >= HOLD_SECONDS):
                        phrase.append(pending_letter)
                        hold_start     = None
                        last_letter    = None
                        pending_letter = None

                elif not show_final:
                    hold_start     = None
                    last_letter    = None
                    pending_letter = None

                # ui

                # letter track
                if pending_letter and not show_final and result.multi_hand_landmarks:
                    hl = result.multi_hand_landmarks[0]
                    cx = int(hl.landmark[0].x * w)
                    cy = int(hl.landmark[0].y * h)
                    conf_color = (0, 255, 80) if pending_conf > 0.85 else (0, 200, 255)
                    cv2.putText(frame, pending_letter,
                                (cx - 90, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, conf_color, 3)
                    cv2.putText(frame, f"{int(pending_conf * 100)}%",
                                (cx - 90, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)

                #  progress bar
                if hold_start and pending_letter:
                    elapsed  = now - hold_start
                    progress = min(elapsed / HOLD_SECONDS, 1.0)
                    bar_w    = int(w * progress)
                    cv2.rectangle(frame, (0, h - 20), (bar_w, h), (180, 0, 255), -1)
                    cv2.putText(frame, f"Holding '{pending_letter}'  {int(progress * 100)}%",
                                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

                # phrase buffer
                phrase_str = "".join(phrase) if phrase else "(start signing)"
                cv2.rectangle(frame, (0, h - 90), (w, h - 22), (20, 20, 20), -1)
                cv2.putText(frame, phrase_str,
                            (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # final phrase display
                if show_final:
                    cv2.putText(frame, final_phrase,
                                (10, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 120), 3)

            
            cv2.imshow("Hand Recognition | ASL + Counting", frame)

            if key == ord('q'):
                break
            elif key == 9:  # TAB
                mode = MODE_ASL if mode == MODE_COUNTING else MODE_COUNTING
                phrase = []; hold_start = None; pending_letter = None
                show_final = False
            elif key == ord('c'):
                phrase = []; hold_start = None; pending_letter = None
                show_final = False; final_phrase = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
