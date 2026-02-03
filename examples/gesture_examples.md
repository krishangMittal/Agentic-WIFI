
# Gesture-Based Agentic Actions - Examples

## 🎮 How It Works

```
Physical Gesture → WiFi Disruption → Spectrogram → Classifier → LLM → Smart Action
```

---

## 💡 Smart Home Control

### Turn On Lights
**Gesture:** Wave left hand (A17)

```python
from src.gesture_agent import GestureAgent

agent = GestureAgent()

# Evening: Warm lights
agent.execute_gesture_action('A17', confidence=0.92, 
                             additional_context={'time': 'evening'})
# → "[SMART HOME] Turning on warm lights (evening mode)"

# Night: Dim lights
agent.execute_gesture_action('A17', confidence=0.92,
                             additional_context={'time': 'night'})
# → "[SMART HOME] Turning on dim lights (night mode)"
```

### Turn Off Lights
**Gesture:** Wave right hand (A18)

```python
agent.execute_gesture_action('A18', confidence=0.88)
# → "[SMART HOME] Turning off lights"
```

---

## 🎵 Media Control

### Volume Up
**Gesture:** Raise left hand (A13)

```python
agent.execute_gesture_action('A13', confidence=0.85,
                             additional_context={'music_playing': True})
# → "[MEDIA] Volume +20%"
```

### Volume Down
**Gesture:** Raise right hand (A14)

```python
agent.execute_gesture_action('A14', confidence=0.87,
                             additional_context={'music_playing': True})
# → "[MEDIA] Volume -20%"
```

### Next Track
**Gesture:** Throw right (A21)

```python
agent.execute_gesture_action('A21', confidence=0.90)
# → "[MEDIA] Skipping to next track"
```

### Previous Track
**Gesture:** Throw left (A20)

```python
agent.execute_gesture_action('A20', confidence=0.89)
# → "[MEDIA] Going to previous track"
```

---

## 🚨 Emergency Detection

### Fall Detection
**Gesture:** Sudden bend down (A19 or A27)

```python
# Regular user
agent.execute_gesture_action('A19', confidence=0.75)
# → "[INFO] Normal bending motion detected"

# Elderly person
agent.execute_gesture_action('A19', confidence=0.75,
                             additional_context={
                                 'user_age': 75,
                                 'elderly_mode': True,
                                 'alone_at_home': True
                             })
# → "[!] ALERT: Monitoring for fall recovery (elderly mode)"

# No recovery after 3 seconds
agent.execute_gesture_action('A27', confidence=0.82,
                             additional_context={
                                 'elderly_mode': True,
                                 'no_upward_motion_seconds': 3
                             })
# → "[!!!] CRITICAL: Calling emergency services"
```

---

## 🏋️ Exercise & Rehabilitation

### Start Workout
**Gesture:** Stretching (A01)

```python
agent.execute_gesture_action('A01', confidence=0.88)
# → "[FITNESS] Workout mode activated"
```

### Log Exercise
**Gesture:** Jumping (A26)

```python
agent.execute_gesture_action('A26', confidence=0.91,
                             additional_context={'workout_active': True})
# → "[FITNESS] Logged exercise: jumping"
```

### Count Reps
**Gesture:** Limb extension (A11)

```python
agent.execute_gesture_action('A11', confidence=0.86,
                             additional_context={'rep_count': 5})
# → "[FITNESS] Rep 5/10 complete"
```

---

## 🧠 LLM-Powered Intelligence

### Context-Aware Decision Making

The LLM (DeepSeek) makes the system **smart**, not just rule-based:

#### Example 1: Same Gesture, Different Context

```python
# Morning in bedroom
agent.execute_gesture_action('A17', confidence=0.90,
                             additional_context={
                                 'time': 'morning',
                                 'room': 'bedroom'
                             })
# LLM decides: "Morning + bedroom → Turn on bright lights for waking up"

# Evening in living room
agent.execute_gesture_action('A17', confidence=0.90,
                             additional_context={
                                 'time': 'evening',
                                 'room': 'living_room'
                             })
# LLM decides: "Evening + living room → Turn on warm ambient lights"
```

#### Example 2: Ambiguous Gesture

```python
# User raises hand - but doing what?
agent.execute_gesture_action('A13', confidence=0.65,
                             additional_context={
                                 'music_playing': True,
                                 'tv_on': False
                             })
# LLM: "Music playing + hand raise → Increase volume"

agent.execute_gesture_action('A13', confidence=0.65,
                             additional_context={
                                 'music_playing': False,
                                 'presenting': True
                             })
# LLM: "Presenting mode + hand raise → Asking question, no action"
```

#### Example 3: Emergency Triage

```python
# Bending down
agent.execute_gesture_action('A19', confidence=0.70,
                             additional_context={
                                 'user_age': 30,
                                 'fitness_tracker_active': True
                             })
# LLM: "Young user + fitness active → Normal exercise, log it"

agent.execute_gesture_action('A19', confidence=0.70,
                             additional_context={
                                 'user_age': 80,
                                 'alone': True,
                                 'last_motion_10_min_ago': True
                             })
# LLM: "Elderly + alone + infrequent motion → POTENTIAL FALL, monitor closely"
```

---

## 🛠️ Custom Actions

### Add Your Own Gestures

Edit `config/gesture_actions.yaml`:

```yaml
gestures:
  A17:  # Wave left
    action: "turn_on_lights"
    description: "Turn on lights"
    context_rules:
      - if: "room_is_kitchen"
        action: "turn_on_coffee_maker"  # ← Custom!
      - if: "morning_routine"
        action: "start_news_briefing"    # ← Custom!
```

Then use it:

```python
agent.execute_gesture_action('A17', confidence=0.92,
                             additional_context={
                                 'room': 'kitchen',
                                 'time': 'morning'
                             })
# → "[SMART HOME] Starting coffee maker + news briefing"
```

---

## 🎯 Real-World Use Cases

### 1. Elderly Care
```python
# Continuous monitoring
while True:
    gesture = detect_gesture_from_wifi()  # Your MM-Fi pipeline
    
    if gesture in ['A19', 'A27']:  # Bending/bowing
        result = agent.execute_gesture_action(
            gesture,
            confidence=0.80,
            additional_context={
                'elderly_mode': True,
                'monitoring_enabled': True
            }
        )
        
        if "CRITICAL" in result:
            call_emergency()
```

### 2. Hands-Free Smart Home
```python
# No buttons, no voice - just gestures!
gesture_map = {
    'A17': 'lights_on',
    'A18': 'lights_off',
    'A13': 'volume_up',
    'A14': 'volume_down',
    'A20': 'previous',
    'A21': 'next'
}

for gesture_code, action_name in gesture_map.items():
    agent.execute_gesture_action(gesture_code, confidence=0.85)
```

### 3. Accessibility
```python
# For users with limited mobility or speech
agent.execute_gesture_action('A17', confidence=0.75,
                             additional_context={
                                 'accessibility_mode': True,
                                 'user_needs': 'speech_impaired'
                             })
# LLM: "Accessibility mode → Convert gesture to text-to-speech command"
```

### 4. Fitness Tracking
```python
# Automatic workout logging
exercise_gestures = ['A01', 'A06', 'A11', 'A12', 'A26']

for gesture in exercise_gestures:
    agent.execute_gesture_action(gesture, confidence=0.88,
                                 additional_context={
                                     'workout_session_id': 'session_123',
                                     'track_calories': True
                                 })
```

---

## 🚀 Next Steps

1. **Download MM-Fi data** - Get real WiFi CSI samples
2. **Train classifier** - Recognize your 27 gestures
3. **Customize actions** - Edit `gesture_actions.yaml`
4. **Test end-to-end** - `python src/gesture_agent.py`
5. **Deploy** - Connect to your smart home APIs

---

## 📚 Files

- **Config:** `config/gesture_actions.yaml`
- **Agent:** `src/gesture_agent.py`
- **Demo:** `python src/gesture_agent.py`
- **Setup:** `docs/MMFI_SETUP.md`

**Your "Siri for WiFi" is now a gesture-controlled smart assistant!** 🎉

