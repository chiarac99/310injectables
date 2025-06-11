/* ----------- Pin assignment ----------- */
#define stepPin1 8
#define dirPin1  9
#define stepPin2 11
#define dirPin2  12
#define limitSwitch1 6
#define limitSwitch2 7

/* ----------- Motion parameters ----------- */
#define steps1 1000                  // Motor-1 outward-stroke steps
#define steps2 2200                  // Motor-2 outward-stroke steps
#define TOTAL_DURATION 2100000UL     // 5 000 000 µs = 5 s outward stroke

#define SNAP_DELAY_MICROS  100       // delay before printing “SNAP”
#define REVERSE_PAUSE_MS   500       // pause after both switches hit

/* ----------- Direction definitions ----------- */
const bool DIR_TOWARD_HOME = HIGH;
const bool DIR_AWAY_HOME   = LOW;

/* ----------- Timing variables (filled in setup) ----------- */
unsigned long interval1, interval2;
unsigned long lastStepTime1 = 0, lastStepTime2 = 0;
unsigned long pulseStart1   = 0, pulseStart2   = 0;
bool          pulseHigh1 = false, pulseHigh2 = false;

/* ----------- Counters & flags ----------- */
int  currentStep1 = 0, currentStep2 = 0;
bool homed1 = false, homed2 = false, systemHomed = false;
bool forwardStroke = true;      // “away from switches”
bool snapSent = false;

/* ========================================================= */
void setup() {
  pinMode(stepPin1, OUTPUT);   pinMode(dirPin1,  OUTPUT);
  pinMode(stepPin2, OUTPUT);   pinMode(dirPin2,  OUTPUT);
  pinMode(limitSwitch1, INPUT_PULLUP);
  pinMode(limitSwitch2, INPUT_PULLUP);

  Serial.begin(115200);

  /* compute pulse intervals */
  interval1 = TOTAL_DURATION / steps1;
  interval2 = TOTAL_DURATION / steps2;

  /* start in homing direction */
  digitalWrite(dirPin1, DIR_TOWARD_HOME);
  digitalWrite(dirPin2, DIR_TOWARD_HOME);
}
/* ========================================================= */
void loop() {
  unsigned long now = micros();

/* ---------- 1. HOMING -------------------------------------------------- */
  if (!systemHomed) {
    if (!homed1 && digitalRead(limitSwitch1) == LOW) { homed1 = true; Serial.println("Stepper 1 homed"); }
    if (!homed2 && digitalRead(limitSwitch2) == LOW) { homed2 = true; Serial.println("Stepper 2 homed"); }

    if (!homed1) simpleStep(stepPin1, now, 2000, lastStepTime1, pulseStart1, pulseHigh1); // 5 ms pulse rate
    if (!homed2) simpleStep(stepPin2, now, 2000, lastStepTime2, pulseStart2, pulseHigh2);

    if (homed1 && homed2) {
      systemHomed   = true;
      forwardStroke = true;                 // first stroke goes outward
      snapSent      = false;
      setDirection(forwardStroke);
      resetStroke(now);
      Serial.println("Both motors homed → starting outward stroke");
    }
    return;                                // skip remainder while homing
  }

/* ---------- 2. SNAP on return stroke ----------------------------------- */
  // if (!forwardStroke) {                    // currently moving toward switches
  //   if (!snapSent && digitalRead(limitSwitch1) == LOW) {
  //     delayMicroseconds(SNAP_DELAY_MICROS);
  //     Serial.println("SNAP");
  //     snapSent = true;
  //   }
  // }

/* ---------- 3. MOTION CONTROL ------------------------------------------ */
  if (forwardStroke) {                     // outward stroke: synced stepping
    syncStep(stepPin1, now, interval1, currentStep1, steps1,
             lastStepTime1, pulseStart1, pulseHigh1);
    syncStep(stepPin2, now, interval2, currentStep2, steps2,
             lastStepTime2, pulseStart2, pulseHigh2);
  } else {                                 // return stroke
    // Motor-1: keep stepping at same speed until its switch clicks
    if (digitalRead(limitSwitch1) != LOW)
      syncStepNoCount(stepPin1, now, interval1,
                      lastStepTime1, pulseStart1, pulseHigh1);
    else                                    // ensure final LOW if pulse still HIGH
      finalizePulse(stepPin1, now, pulseStart1, pulseHigh1);

    // Motor-2: same logic
    if (digitalRead(limitSwitch2) != LOW)
      syncStepNoCount(stepPin2, now, interval2,
                      lastStepTime2, pulseStart2, pulseHigh2);
    else
      finalizePulse(stepPin2, now, pulseStart2, pulseHigh2);
  }

/* ---------- 4. OUTWARD stroke finished → start return ------------------ */
  if ( forwardStroke &&
       currentStep1 >= steps1 && currentStep2 >= steps2 &&
       !pulseHigh1 && !pulseHigh2 )
  {
    forwardStroke = false;                 // switch to return stroke
    snapSent = false;
    setDirection(forwardStroke);           // DIR_TOWARD_HOME
    Serial.println("Outward stroke complete → return stroke");
  }

/* ---------- 5. Both switches LOW → pause & start next outward ---------- */
  if ( !forwardStroke &&
       digitalRead(limitSwitch1) == LOW && digitalRead(limitSwitch2) == LOW )
  { 
    //delay(REVERSE_PAUSE_MS);
    Serial.println("SNAP");
    snapSent = true;
    forwardStroke = true;
    //snapSent = false;
    setDirection(forwardStroke);           // DIR_AWAY_HOME
    resetStroke(now);
    Serial.println("Return stroke complete → next outward stroke");
  }
}
/* ====================================================================== */
/* -------- Helper: synced step WITH counter (outward stroke) ----------- */
void syncStep(int pin, unsigned long now, unsigned long interval,
              int &stepCount, int maxSteps,
              unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh)
{
  if (!pulseHigh && stepCount < maxSteps && now - lastTime >= interval) {
    digitalWrite(pin, HIGH);
    pulseStart = now;
    pulseHigh  = true;
    lastTime   = now;
    stepCount++;
  } else if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}
/* -------- Helper: synced step NO counter (return stroke) -------------- */
void syncStepNoCount(int pin, unsigned long now, unsigned long interval,
                     unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh)
{
  if (!pulseHigh && now - lastTime >= interval) {
    digitalWrite(pin, HIGH);
    pulseStart = now;
    pulseHigh  = true;
    lastTime   = now;
  } else if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}
/* -------- Helper: one-size-fits homing / slow stepping ---------------- */
void simpleStep(int pin, unsigned long now, unsigned long interval,
                unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh)
{
  if (!pulseHigh && now - lastTime >= interval) {
    digitalWrite(pin, HIGH);
    pulseStart = now;
    pulseHigh  = true;
    lastTime   = now;
  } else if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}
/* -------- Helper: ensure pulse finishes LOW --------------------------- */
void finalizePulse(int pin, unsigned long now,
                   unsigned long &pulseStart, bool &pulseHigh)
{
  if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}
/* -------- DIR-pin helper --------------------------------------------- */
void setDirection(bool awayFromSwitch) {
  digitalWrite(dirPin1, awayFromSwitch ? DIR_AWAY_HOME : DIR_TOWARD_HOME);
  digitalWrite(dirPin2, awayFromSwitch ? DIR_AWAY_HOME : DIR_TOWARD_HOME);
}
/* -------- Reset counters & timers for a new outward stroke ------------ */
void resetStroke(unsigned long now) {
  currentStep1 = currentStep2 = 0;
  lastStepTime1 = lastStepTime2 = now;
}