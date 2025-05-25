/*
main.ino

Main Arduino file for 310 Injectables

This file should handle all comms with electronics. These include:
- limit switches for linear actuator positioning
- Nema 17 motors for syringe alignment subsystem and two linear actuators

*/

#define DIR_PIN_ALIGN 12 // linear actuator that moves aligning steps up and down
#define DIR_PIN_CUT 9 // linear actuator that moves cutting step up and down
#define STEP_PIN_ALIGN 11 
#define STEP_PIN_CUT 8

#define LIMIT_SWITCH_ALIGN 4 // limit switch for positioning linear actuator 1
#define LIMIT_SWITCH_CUT 2 // limit switch for positioning linear actuator 2

/*
LINEAR ACTUATOR POSITIONING
- linear actuator position is in integer number of steps from the zero position
- the zero position is the position of the limit switch at the top its travel
- it should hit this point once every cycle (when going up)
*/
#define microStepsPerRotation 200 // 5mm pitch of lead screw
const int StepsPerAlignCycle = int(microStepsPerRotation*5.08); // number of steps to do at least 1" of travel (from zero position) - 5.08 rotations
const int StepsPerCutCycle = int(microStepsPerRotation*10.16); // number of steps to do at least 2" of travel - 10.16 rotations

#define actuatorUpDir true
#define actuatorDownDir false
bool direction = actuatorUpDir; // start direction upwards
uint8_t alignActivated = 1; // is the actuator active for this loop
uint8_t cutActivated = 1;
#define ActuatorStepDelay 500  // delay btw HIGH and LOW signal for align and cutting stepper motors

int alignStepCount = 0; // step count record
int cutStepCount = 0;

// REPOSITION PADDLE
// variables received from python for repositioning
#define LIMIT_SWITCH_PADDLE 13
#define DIR_PIN_PADDLE 6
#define STEP_PIN_PADDLE 7
#define HOMING_INTERVAL_PADDLE 2500 // microseconds/step
#define FAST_INTERVAL_PADDLE 1250 // microseconds/step
char syringeDir;  // dir of syringe (determines where to send paddle before repositioning)
int syringeCut;  // position in pixels of where to cut
int paddlePos = 0;  // saves current position of paddle
unsigned long lastStepTimePaddle = 0;
unsigned long pulseStartPaddle = 0;
bool pulseHighPaddle = false;
int currentStepPaddle = 0;

#define PADDLE_TO_HOME 0
#define PADDLE_AWAY_FROM_HOME 1
#define PADDLE_OFF 2
int directionPaddle = PADDLE_OFF; //PADDLE_TO_HOME;  // start in limitswitch dir
int lastDirectionPaddle = PADDLE_OFF;
int intervalPaddle = HOMING_INTERVAL_PADDLE;
int targetPaddlePos = 0;
uint8_t paddleTargetReached = 1; // bool for state machine to know when a target has been reached

#define PADDLE_HOME_POS 0
#define PADDLE_AWAY_FROM_HOME_POS 500 // 5 x 200 steps (1 x rev)

// states
typedef enum {
  INITIALISING_CAMERA,
  INITIALISING_BLADE,
  HOMING_PADDLE,
  MOVING_UP, // no need to home actuators because we will always be moving upwards first; we check if we reach the limit switch at the top of EVERY cycle
  MOVING_DOWN,
  ASK_FOR_IMG,
  WAITING_FOR_IMG_OUTPUT,
  PRE_POSITION_PADDLE,
  TESTING_REPOSITIONING
} systemState_t;
systemState_t systemState = ASK_FOR_IMG;

void setup() {
  Serial.begin(115200);
  while (Serial.available()) Serial.read();  // Clear buffer

  // linear actuators
  pinMode(DIR_PIN_ALIGN, OUTPUT);
  pinMode(DIR_PIN_CUT, OUTPUT);
  pinMode(STEP_PIN_ALIGN, OUTPUT);
  pinMode(STEP_PIN_CUT, OUTPUT);

  digitalWrite(DIR_PIN_ALIGN, direction); // initialise dir of align actuator
  digitalWrite(DIR_PIN_CUT, direction); // initialise dir of cut actuator

  // limitswitches
  pinMode(LIMIT_SWITCH_ALIGN, INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_CUT, INPUT_PULLUP);

  // paddle
  pinMode(DIR_PIN_PADDLE, OUTPUT);
  pinMode(STEP_PIN_PADDLE, OUTPUT);
  digitalWrite(DIR_PIN_PADDLE, PADDLE_TO_HOME);

  delay(2000);

}

void loop() {


  // update state
  stateHandler();

  // move step/cut actuators if appropriate
  moveActuators();

  // move paddle if appropriate
  updatePaddleDir();
  movePaddle();
}

void stateHandler(){
  // Serial.print("target pos: "); Serial.println(targetPaddlePos);

  // Serial.print("real pos: "); Serial.println(paddlePos);

  // state machine
  switch (systemState) {

      case INITIALISING_CAMERA:
        // let python know that we're ready to start the camera
        // may need two states bc then we need to wait for python to tell us okay
        break;

      case INITIALISING_BLADE:
        // turn on motor for rotary saw
        break;

      case HOMING_PADDLE:
        // move paddle to limit switch
        // if paddle limitswitch activated, move to next step + reset paddlePos = 0
        if (paddleHomed()){
          paddlePos = PADDLE_HOME_POS; // set paddle position to home
          Serial.println("paddled home!");
          // while(1);
        }

        break;

      case MOVING_UP:
        // move actuators up until both activate top limit switches
        if (movedUp()) systemState = ASK_FOR_IMG;
        break;

      case MOVING_DOWN:
        // move actuators down until both steppers do the correct number of StepsPerCycle
        if (movedDown()) {
          Serial.println("moving back up!");
          // reactivated actuators
          alignActivated = 1;
          cutActivated = 1;

          // restart step counts
          alignStepCount = 0;
          cutStepCount = 0;

          // change state
          systemState = MOVING_UP;
        }
        break;

      case ASK_FOR_IMG:
        Serial.println("SNAP"); // send msg to python to take image
        systemState = WAITING_FOR_IMG_OUTPUT;
        break;

      case WAITING_FOR_IMG_OUTPUT:
        if (receivedImgOutput()) {

          // check input
          if (syringeDir == 'R' || syringeDir == 'L'){
          // activate steppers
          // alignActivated = 1;
          // cutActivated = 1;

            // change state
            setPaddlePreposition();
            intervalPaddle = FAST_INTERVAL_PADDLE; // also increase speed
            systemState = PRE_POSITION_PADDLE;

          } else {
            // go back to SNAP
            systemState = ASK_FOR_IMG;
          }
        }
        break;

      case PRE_POSITION_PADDLE:
        if (paddleTargetReached){

          // change state
          setPaddlePlungerPos();
          systemState = TESTING_REPOSITIONING;

        }
        break;

      case TESTING_REPOSITIONING:
        if (paddleTargetReached){
          
          // change state
          systemState = ASK_FOR_IMG;

        }
        break;

      default:

        // statements
        Serial.println("this is not a state :'(");
        break;

    }


}


/* STATE FUNCTIONS */

uint8_t paddleHomed(){
  // // return true if the paddle limitswitch is activated
  uint8_t homed = limitswitchActivated(LIMIT_SWITCH_PADDLE);
  
  if (homed){
    return 1;
  }

  return 0;

}

uint8_t movedUp(){
  // return true if both limitswitches have been activated

  // bools for recording whether alignment and cutting steps have reached the end of either their upward or downward cycle
  uint8_t aligned = limitswitchActivated(LIMIT_SWITCH_ALIGN);
  uint8_t cut = limitswitchActivated(LIMIT_SWITCH_CUT);

  // set dir
  direction = actuatorUpDir;

  // deactivate the stepper if limitswitches were activated
  if (aligned) alignActivated = 0; 
  if (cut) cutActivated = 0;

  // return true if both actuators have reached their endpoints
  if (!alignActivated && !cutActivated) return 1;

  // otherwise keep moving align actuator up
  return 0;
}

uint8_t movedDown(){
  // return true if completed the correct number of steps for a cycle
  
  // set dir
  direction = actuatorDownDir;

  // get out if both actuators are deactivated
  if (!alignActivated && !cutActivated) return 1;

  // deactivate actuators if step count reached
  if (alignStepCount >= StepsPerAlignCycle) alignActivated = 0;
  if (cutStepCount >= StepsPerCutCycle) cutActivated = 0;

  // increment step count
  alignStepCount++;
  cutStepCount+=2; // cutting step is moving twice as fast as align step

  return 0;

}

void moveActuators(void){
  // update direction
  digitalWrite(DIR_PIN_ALIGN, direction);
  digitalWrite(DIR_PIN_CUT, direction);

  // update step
  if (alignActivated || cutActivated) {
    // Step 1: Leading edge
    if (alignActivated) digitalWrite(STEP_PIN_ALIGN, HIGH);
    if (cutActivated)   digitalWrite(STEP_PIN_CUT, HIGH);
    delayMicroseconds(ActuatorStepDelay);

    // Step 2: Trailing edge
    digitalWrite(STEP_PIN_ALIGN, LOW);
    digitalWrite(STEP_PIN_CUT, LOW);
    delayMicroseconds(ActuatorStepDelay);

    // Step 3: Extra step for cut actuator to make it 2x faster
    if (cutActivated) {
      digitalWrite(STEP_PIN_CUT, HIGH);
      delayMicroseconds(ActuatorStepDelay);
      digitalWrite(STEP_PIN_CUT, LOW);
      delayMicroseconds(ActuatorStepDelay);
    }
  }
  // if neither are activated, do nothing

}


void setPaddlePreposition(void){
  // sets target position of paddle with known syringe dir
  if (syringeDir == 'R'){
    // move paddle to home
    directionPaddle = PADDLE_TO_HOME;
    targetPaddlePos = PADDLE_HOME_POS;
  }
  else if (syringeDir == 'L'){
    // move paddle to opposite side
    directionPaddle = PADDLE_AWAY_FROM_HOME;
    targetPaddlePos = PADDLE_AWAY_FROM_HOME_POS;
  }
  else {
    Serial.println("ERROR: not a syringe dir");
  }

  // set the target flag false
  paddleTargetReached = 0;
}


void setPaddlePlungerPos(void){
  // if (syringeDir == 'R'){

  // } else if (syringeDir == 'L'){

  // } else {
  //   Serial.println("ERROR: not a syringe dir");
  // }

  // map pixel space to paddle motor space (i.e. number of steps to send)
  // 

  // set dir
  if (syringeDir == 'R'){
    // move paddle away from home
    directionPaddle = PADDLE_AWAY_FROM_HOME;
  }
  else if (syringeDir == 'L'){
    // move paddle towards home
    directionPaddle = PADDLE_TO_HOME;
  }
  else {
    Serial.println("ERROR: not a syringe dir");
  }

  // python should send a value that has already been mapped to our paddle space
  // since the paddle space is not changing
  // python accounts for slight changes in camera positioning
  // and always sends a value equivalent to the number of steps needed to travel by the paddle stepper
  // this is a value from 0 (HOME) to MAX
  targetPaddlePos = syringeCut;

  // set the target flag false
  paddleTargetReached = 0;

}


uint8_t movedPaddleToTarget(){
  // check if moved to target position
  // // steps are impossible to miss
  // if (paddlePos == targetPaddlePos) return 1;
  
  if (directionPaddle == PADDLE_TO_HOME){
    // if moving towards home, paddlePos will have to be less than target position to pass target
    if (paddlePos <= targetPaddlePos) {
      paddleTargetReached = 1;
      return 1;
    }
  } else if (directionPaddle == PADDLE_AWAY_FROM_HOME) {
    // paddle is moving away from home, so paddlePos will have to be larger than target to pass target
    if (paddlePos >= targetPaddlePos){
      paddleTargetReached = 1;
      return 1;
    }
  } else if (directionPaddle == PADDLE_OFF){
    // don't change the paddleTargetReached flag
    // but still still return true
    // this addresses both the case that paddle if OFF, but a target has just been set
    // or that paddle is off and no target exists
    return 1;
  } else {
    Serial.println("ERROR: Not a valid paddle dir");
  }


  return 0;
}

void updatePaddleDir(void){
  
  if ((directionPaddle != PADDLE_OFF) && (directionPaddle != lastDirectionPaddle)) {
    digitalWrite(DIR_PIN_PADDLE, directionPaddle);
    lastDirectionPaddle = directionPaddle;
  }
}

void movePaddle(void){
  if (directionPaddle == PADDLE_OFF) {
    return; // Paddle is inactive
  }

  // 🚫 Don't move if already at target
  if (movedPaddleToTarget()) {
    directionPaddle = PADDLE_OFF;
    return;
  }
  
  // update step
  unsigned long now = micros();
  if (!pulseHighPaddle && (now - lastStepTimePaddle >= intervalPaddle)) {
    // next step to be sent according to time interval wanted (i.e. set speed)
    // pull step pin HIGH
    digitalWrite(STEP_PIN_PADDLE, HIGH);
    pulseStartPaddle = now;
    pulseHighPaddle = true;
    lastStepTimePaddle = now;

    // update paddle pos
    if (directionPaddle == PADDLE_TO_HOME){
      // this is the homing dir
      // so should go down to 0 (which is home)
      paddlePos--;
    } else {
      // going away from home
      // increase pos
      paddlePos++;
    }
  } else if (pulseHighPaddle && (now - pulseStartPaddle >= 10)) {
    // pull step pin LOW to send falling edge
    digitalWrite(STEP_PIN_PADDLE, LOW);
    pulseHighPaddle = false;
  }


}


uint8_t limitswitchActivated(uint8_t pin){
  // return true if limit switch at specified pin is activated
  if (!digitalRead(pin)){
    return 1;
  }
  return 0;
}


uint8_t receivedImgOutput(void) {
  // output from python is going to be in the form:
  // <d[char]c[int]>
  // where d is direction the syringe is pointing: 'L' or 'R'
  // and c is where to cut in pixels (int)

  String buf = "";
  static uint8_t data_packet_started = 0;
  static uint8_t done = 0;

  while (Serial.available() && !done) {
    char inChar = (char)Serial.read();

    if (data_packet_started) {
        // add to buffer
        // include end char
        buf += inChar;
      if (inChar == '>') {
        // received end char
        // stop reading serial
        done = 1;
        Serial.println(buf);
      }
    } else {
      if (inChar == '<') {
        // received start char
        // start reading serial
        data_packet_started = 1;
      }
    }
  }

  if (done) {
    // Try parsing manually without exceptions
    int dIdx = buf.indexOf("d");
    int cIdx = buf.indexOf("c");

    if (dIdx != -1 && cIdx != -1) {
      syringeDir = buf.charAt(dIdx + 1);
      
      int startC = cIdx + 1;
      int endC = buf.indexOf('>');
      if (endC != -1) {
        String cStr = buf.substring(startC, endC);
        syringeCut = cStr.toInt();
        
        Serial.print("dir: ");
        Serial.println(syringeDir);
        Serial.print("cut: ");
        Serial.println(syringeCut);
      } else {
        Serial.println("Error: could not find closing bracket for c");
      }
    } else {
      Serial.println("Error: d or c not found");
    }

    buf = "";
    done = 0;
    data_packet_started = 0;

    return 1;
  }

  return 0;
}
