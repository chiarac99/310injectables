/* ----------- Pin assignment ----------- */
#define stepPin1 13
#define dirPin1  12
#define stepPin2 10
#define dirPin2  9
#define stepPin3 7
#define dirPin3  6
#define limitSwitch1 4
#define limitSwitch2 3
#define limitSwitchL 11
#define limitSwitchR 8

/* ----------- Motion parameters ----------- */
#define steps1 1800 // This value represents the delta_h for moving steps. On 200 steps/rev setting, 1000 steps == 1 inch
#define steps2 2200 // This value represents the delta_h for cutting step. On 200 steps/rev setting, 1000 steps == 1 inch
#define TOTAL_DURATION 3000000UL // Half a stroke duration 1000000 == 1 sec 
#define SNAP_DELAY_MICROS  100 // Time gap between SNAP sent and limit switches triggered

/* ----------- Direction definitions ----------- */
const bool DIR_TOWARD_HOME = HIGH; // For both linear actuators, HIGH is towards home
const bool DIR_AWAY_HOME   = LOW;

/* ----------- Timing variables ----------- */
unsigned long interval1, interval2; // Interval based on steps and duration
unsigned long lastStepTime1 = 0, lastStepTime2 = 0, lastStepTime3 = 0; 
unsigned long pulseStart1 = 0, pulseStart2 = 0, pulseStart3 = 0;
bool pulseHigh1 = false, pulseHigh2 = false, pulseHigh3 = false;

/* ----------- Counters & flags ----------- */
int currentStep1 = 0, currentStep2 = 0;
bool homed1 = false, homed2 = false, systemHomed = false;
bool forwardStroke = true;
bool snapSent = false;
bool motor3Homed = false;

/* ----------- Serial control variables ----------- */
char orientation = 'L';
int cutSteps = 0;
int cutStepCount = 0;
bool newCommandAvailable = false;

/* ========================================================= */
void setup() {
  pinMode(stepPin1, OUTPUT); pinMode(dirPin1, OUTPUT);
  pinMode(stepPin2, OUTPUT); pinMode(dirPin2, OUTPUT);
  pinMode(stepPin3, OUTPUT); pinMode(dirPin3, OUTPUT);

  pinMode(limitSwitch1, INPUT_PULLUP);
  pinMode(limitSwitch2, INPUT_PULLUP);
  pinMode(limitSwitchL, INPUT_PULLUP);
  pinMode(limitSwitchR, INPUT_PULLUP);

  Serial.begin(115200);

  interval1 = TOTAL_DURATION / steps1;
  interval2 = TOTAL_DURATION / steps2;

  digitalWrite(dirPin1, DIR_TOWARD_HOME);
  digitalWrite(dirPin2, DIR_TOWARD_HOME);
}

void loop() {
  unsigned long now = micros();

  // Serial input parsing
  while (Serial.available()) {
    static String input = "";
    char ch = Serial.read();
    if (ch == '<') input = "";
    input += ch;
    if (ch == '>') {
      int dIndex = input.indexOf('d');
      int cIndex = input.indexOf('c');
      if (dIndex != -1 && cIndex != -1) {
        orientation = input.charAt(dIndex + 1);
        Serial.println(orientation);
        cutSteps = input.substring(cIndex + 1, input.length() - 1).toInt();
        Serial.println(cutSteps);
        newCommandAvailable = true;
      }
      input = "";
    }
  }

  // Homing
  if (!systemHomed) {
    if (!homed1 && digitalRead(limitSwitch1) == LOW) { homed1 = true; Serial.println("Stepper 1 homed"); }
    if (!homed2 && digitalRead(limitSwitch2) == LOW) { homed2 = true; Serial.println("Stepper 2 homed"); }

    if (!homed1) simpleStep(stepPin1, now, 2000, lastStepTime1, pulseStart1, pulseHigh1);
    if (!homed2) simpleStep(stepPin2, now, 2000, lastStepTime2, pulseStart2, pulseHigh2);

    if (homed1 && homed2) {
      systemHomed = true;
      forwardStroke = true;
      snapSent = false; 
      setDirection(forwardStroke);
      resetStroke(now);
      Serial.println("Both motors homed → starting outward stroke");
    }
    return;
  }

  // Motor 3 cutting motion runs in parallel
  if (snapSent) {
    syncStep(stepPin3, now, 1500, cutStepCount, cutSteps,
             lastStepTime3, pulseStart3, pulseHigh3);
    //Serial.println(cutSteps);
    //Serial.println(cutStepCount);
    if (cutStepCount >= cutSteps && !pulseHigh3) {
      snapSent = false;
      Serial.println("Motor 3 cutting complete");
    }
  }

  // Motion control
  if (forwardStroke) {
    syncStep(stepPin1, now, interval1, currentStep1, steps1,
             lastStepTime1, pulseStart1, pulseHigh1);
    syncStep(stepPin2, now, interval2, currentStep2, steps2,
             lastStepTime2, pulseStart2, pulseHigh2);
  } else {
    if (digitalRead(limitSwitch1) != LOW)
      syncStepNoCount(stepPin1, now, interval1, lastStepTime1, pulseStart1, pulseHigh1);
    else
      finalizePulse(stepPin1, now, pulseStart1, pulseHigh1);

    if (digitalRead(limitSwitch2) != LOW)
      syncStepNoCount(stepPin2, now, interval2, lastStepTime2, pulseStart2, pulseHigh2);
    else
      finalizePulse(stepPin2, now, pulseStart2, pulseHigh2);

    // Motor 3 direction motion to limit switch
    if (!motor3Homed) {
      if (orientation == 'L') {
        digitalWrite(dirPin3, HIGH);
        if (digitalRead(limitSwitchR) != LOW) {
          simpleStep(stepPin3, now, 1500, lastStepTime3, pulseStart3, pulseHigh3);
        } else {
          finalizePulse(stepPin3, now, pulseStart3, pulseHigh3);
          motor3Homed = true;
          Serial.println("Homed Right Motor 3");
        }
      } else {
        digitalWrite(dirPin3, LOW);
        if (digitalRead(limitSwitchL) != LOW) {
          simpleStep(stepPin3, now, 1500, lastStepTime3, pulseStart3, pulseHigh3);
        } else {
          finalizePulse(stepPin3, now, pulseStart3, pulseHigh3);
          motor3Homed = true;
          Serial.println("Homed Left Motor 3");
        }
      }
    }
  }

  // Outward stroke finished
  if (forwardStroke &&
      currentStep1 >= steps1 && currentStep2 >= steps2 &&
      !pulseHigh1 && !pulseHigh2)
  {
    forwardStroke = false;
    setDirection(forwardStroke);
    motor3Homed = false;
    cutStepCount = 0;
    Serial.println("Outward stroke complete → return stroke");
  }

  // Return stroke complete
  if (!forwardStroke &&
      digitalRead(limitSwitch1) == LOW && digitalRead(limitSwitch2) == LOW &&
      motor3Homed)
  {
    delayMicroseconds(SNAP_DELAY_MICROS);
    Serial.println("SNAP");

    // Set direction for cutting stroke
    digitalWrite(dirPin3, (orientation == 'L') ? LOW : HIGH);

    cutStepCount = 0;
    lastStepTime3 = now;
    pulseStart3 = 0;
    pulseHigh3 = false;

    snapSent = true;

    // Begin outward stroke immediately
    forwardStroke = true;
    setDirection(forwardStroke);
    resetStroke(micros());
    Serial.println("Return stroke complete → next outward stroke");
  }
}

/* -------- Helper functions ------------------------ */
void syncStep(int pin, unsigned long now, unsigned long interval,
              int &stepCount, int maxSteps,
              unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh)
{
  if (!pulseHigh && stepCount < maxSteps && now - lastTime >= interval) {
    digitalWrite(pin, HIGH);
    pulseStart = now;
    pulseHigh = true;
    lastTime = now;
    stepCount++;
  } else if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}

void syncStepNoCount(int pin, unsigned long now, unsigned long interval,
                     unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh)
{
  if (!pulseHigh && now - lastTime >= interval) {
    digitalWrite(pin, HIGH);
    pulseStart = now;
    pulseHigh = true;
    lastTime = now;
  } else if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}

void simpleStep(int pin, unsigned long now, unsigned long interval,
                unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh)
{
  if (!pulseHigh && now - lastTime >= interval) {
    digitalWrite(pin, HIGH);
    pulseStart = now;
    pulseHigh = true;
    lastTime = now;
  } else if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}

void finalizePulse(int pin, unsigned long now,
                   unsigned long &pulseStart, bool &pulseHigh)
{
  if (pulseHigh && now - pulseStart >= 10) {
    digitalWrite(pin, LOW);
    pulseHigh = false;
  }
}

void setDirection(bool awayFromSwitch) {
  digitalWrite(dirPin1, awayFromSwitch ? DIR_AWAY_HOME : DIR_TOWARD_HOME);
  digitalWrite(dirPin2, awayFromSwitch ? DIR_AWAY_HOME : DIR_TOWARD_HOME);
}

void resetStroke(unsigned long now) {
  currentStep1 = currentStep2 = 0;
  lastStepTime1 = lastStepTime2 = now;
}