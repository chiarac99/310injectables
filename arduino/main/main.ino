/* ----------- Pin assignment ----------- */
#define stepPin1 13
#define dirPin1  12
#define stepPin2 10
#define dirPin2  9
#define stepPin3 6
#define dirPin3  7
#define limitSwitch1 3
#define limitSwitch2 4
#define limitSwitchL 11
#define limitSwitchR 8
#define dcMotorPin 2

/* ----------- Motion parameters ----------- */
#define steps1 1700
#define steps2 2200
#define TOTAL_DURATION 2300000UL
#define SNAP_DELAY_MICROS  100

/* ----------- Direction definitions ----------- */
const bool DIR_TOWARD_HOME = HIGH;
const bool DIR_AWAY_HOME   = LOW;

/* ----------- Timing variables ----------- */
unsigned long interval1, interval2;
unsigned long lastStepTime1 = 0, lastStepTime2 = 0, lastStepTime3 = 0;
unsigned long pulseStart1 = 0, pulseStart2 = 0, pulseStart3 = 0;
bool pulseHigh1 = false, pulseHigh2 = false, pulseHigh3 = false;
unsigned long returnStrokeDoneTime = 0;
unsigned long motor3HomedTime = 0;


/* ----------- New for motor 3 homing delay ----------- */
unsigned long motor3HomingStartTime = 0;
bool readyToHomeMotor3 = false;

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

/* ----------- Control Flags ----------- */
bool isPaused = false;
bool isReady = false;

/* ----------- Debounce timing trackers ----------- */
unsigned long debounceStart1 = 0;
unsigned long debounceStart2 = 0;
unsigned long debounceStartL = 0;
unsigned long debounceStartR = 0;

void setup() {
  Serial.begin(115200);
  pinMode(dcMotorPin, OUTPUT);
  while (!isReady) {
    if (Serial.available()) {
      String command = Serial.readStringUntil('\n');
      command.trim();
      if (command == "READY") {
        isReady = true;
        Serial.println(">> Arduino READY");
        digitalWrite(dcMotorPin, HIGH);
      }
    }
  }

  pinMode(stepPin1, OUTPUT); pinMode(dirPin1, OUTPUT);
  pinMode(stepPin2, OUTPUT); pinMode(dirPin2, OUTPUT);
  pinMode(stepPin3, OUTPUT); pinMode(dirPin3, OUTPUT);

  pinMode(limitSwitch1, INPUT_PULLUP);
  pinMode(limitSwitch2, INPUT_PULLUP);
  pinMode(limitSwitchL, INPUT_PULLUP);
  pinMode(limitSwitchR, INPUT_PULLUP);

  interval1 = TOTAL_DURATION / steps1;
  interval2 = TOTAL_DURATION / steps2;

  digitalWrite(dirPin1, DIR_TOWARD_HOME);
  digitalWrite(dirPin2, DIR_TOWARD_HOME);
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    static String line = "";
    if (ch == '\n') {
      line.trim();
      if (line == "READY") {
        isReady = true;
        Serial.println(">> Arduino READY");
        digitalWrite(dcMotorPin, HIGH);
      } else if (line == "PAUSE") {
        isPaused = true;
        Serial.println(">> Arduino PAUSED");
        digitalWrite(dcMotorPin, LOW);
      } else if (line == "RESUME") {
        isPaused = false;
        Serial.println(">> Arduino RESUMED");
        digitalWrite(dcMotorPin, HIGH);
      }
      line = "";
    } else {
      line += ch;
    }

    static String input = "";
    if (ch == '<') input = "";
    input += ch;
    if (ch == '>') {
      int dIndex = input.indexOf('d');
      int cIndex = input.indexOf('c');
      if (dIndex != -1 && cIndex != -1) {
        orientation = input.charAt(dIndex + 1);
        cutSteps = input.substring(cIndex + 1, input.length() - 1).toInt();
        newCommandAvailable = true;
      }
      input = "";
    }
  }

  if (isPaused) return;

  unsigned long now = micros();

  if (!systemHomed) {
    if (!homed1 && debounceLimitSwitch(limitSwitch1, debounceStart1)) { homed1 = true; Serial.println("Stepper 1 homed"); }
    if (!homed2 && debounceLimitSwitch(limitSwitch2, debounceStart2)) { homed2 = true; Serial.println("Stepper 2 homed"); }

    if (!homed1) simpleStep(stepPin1, now, 3000, lastStepTime1, pulseStart1, pulseHigh1);
    if (!homed2) simpleStep(stepPin2, now, 3000, lastStepTime2, pulseStart2, pulseHigh2);

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

  if (snapSent) {
    bool limitHit = false;
    bool motor3Direction = (orientation == 'L') ? LOW : HIGH;
    if (motor3Direction == LOW) {
      limitHit = debounceLimitSwitch(limitSwitchL, debounceStartL);
    } else {
      limitHit = debounceLimitSwitch(limitSwitchR, debounceStartR);
    }

    if (!limitHit) {
      syncStep(stepPin3, now, 1000, cutStepCount, cutSteps,
               lastStepTime3, pulseStart3, pulseHigh3);
    }

    if ((cutStepCount >= cutSteps && !pulseHigh3) || limitHit) {
      snapSent = false;
      Serial.println("Motor 3 cutting complete");
    }
  }

  if (forwardStroke) {
    syncStep(stepPin1, now, interval1, currentStep1, steps1,
             lastStepTime1, pulseStart1, pulseHigh1);
    syncStep(stepPin2, now, interval2, currentStep2, steps2,
             lastStepTime2, pulseStart2, pulseHigh2);
  } else {
    if (!debounceLimitSwitch(limitSwitch1, debounceStart1))
      syncStepNoCount(stepPin1, now, interval1, lastStepTime1, pulseStart1, pulseHigh1);
    else
      finalizePulse(stepPin1, now, pulseStart1, pulseHigh1);

    if (!debounceLimitSwitch(limitSwitch2, debounceStart2))
      syncStepNoCount(stepPin2, now, interval2, lastStepTime2, pulseStart2, pulseHigh2);
    else
      finalizePulse(stepPin2, now, pulseStart2, pulseHigh2);

    // Add motor 3 homing delay logic
    if (!readyToHomeMotor3 && motor3HomingStartTime > 0 &&
        now - motor3HomingStartTime >= 10000) {
      readyToHomeMotor3 = true;
    }

    if (!motor3Homed && readyToHomeMotor3) {
      if (orientation == 'L') {
        digitalWrite(dirPin3, HIGH);
        if (!debounceLimitSwitch(limitSwitchR, debounceStartR)) {
          simpleStep(stepPin3, now, 1000, lastStepTime3, pulseStart3, pulseHigh3);
        } else {
          finalizePulse(stepPin3, now, pulseStart3, pulseHigh3);
          motor3Homed = true;
          motor3HomedTime = now;
          Serial.println("Homed Right Motor 3");
        }
      } else {
        digitalWrite(dirPin3, LOW);
        if (!debounceLimitSwitch(limitSwitchL, debounceStartL)) {
          simpleStep(stepPin3, now, 1000, lastStepTime3, pulseStart3, pulseHigh3);
        } else {
          finalizePulse(stepPin3, now, pulseStart3, pulseHigh3);
          motor3Homed = true;
          motor3HomedTime = now;
          Serial.println("Homed Left Motor 3");
        }
      }
    }
  }

  if (forwardStroke &&
      currentStep1 >= steps1 && currentStep2 >= steps2 &&
      !pulseHigh1 && !pulseHigh2) {
    forwardStroke = false;
    setDirection(forwardStroke);
    motor3Homed = false;
    cutStepCount = 0;
    motor3HomingStartTime = now;
    readyToHomeMotor3 = false;
    Serial.println("Outward stroke complete → return stroke");
  }

  if (!forwardStroke &&
      debounceLimitSwitch(limitSwitch1, debounceStart1) &&
      debounceLimitSwitch(limitSwitch2, debounceStart2) &&
      motor3Homed &&
      returnStrokeDoneTime == 0) {
    returnStrokeDoneTime = now;
    forwardStroke = true;
    setDirection(forwardStroke);
    resetStroke(now);
    Serial.println("Return stroke complete → next outward stroke");
  }

  if (returnStrokeDoneTime > 0 && now - returnStrokeDoneTime >= SNAP_DELAY_MICROS) {
    Serial.println("SNAP");
    digitalWrite(dirPin3, (orientation == 'L') ? LOW : HIGH);
    cutStepCount = 0;
    lastStepTime3 = now;
    pulseStart3 = 0;
    pulseHigh3 = false;
    snapSent = true;
    returnStrokeDoneTime = 0;
  }
}

/* -------- Helper functions ------------------------ */
void syncStep(int pin, unsigned long now, unsigned long interval,
              int &stepCount, int maxSteps,
              unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh) {
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
                     unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh) {
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
                unsigned long &lastTime, unsigned long &pulseStart, bool &pulseHigh) {
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
                   unsigned long &pulseStart, bool &pulseHigh) {
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

bool debounceLimitSwitch(int pin, unsigned long &startTime) {
  unsigned long debounceDelay = 5000;
  if (digitalRead(pin) == LOW) {
    if (micros() - startTime >= debounceDelay) return true;
  } else {
    startTime = micros();
  }
  return false;
}
