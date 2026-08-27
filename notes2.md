# Classification

## What is Classification?

Classification is a supervised machine learning task where the model predicts a discrete category or class from an input.

For MineArt, classification is used to determine **what is present in a Minecraft image**.

---

## 1. Object Classification

The model identifies objects, animals, mobs, blocks, or other entities in an image.

Examples:

- Cow
- Pig
- Sheep
- Villager
- Zombie
- Dragon Egg
- End Crystal
- Tree
- Flower

Example output:

```text
Cow: 92%
Pig: 4%
Sheep: 2%
Other: 2%
```

---

## Regression


## What is Regression?

Regression is a supervised machine learning task where the model predicts a **continuous numerical value** rather than a discrete class.

For MineArt, regression can be used to estimate the **composition of a Minecraft image**.

---

## 1. Scene Composition Regression

The model can estimate what percentage of the image belongs to different visual components.

Example:

```text
Water:    35%
Land:     20%
Sky:      40%
Objects:   5%
```

---

## 2. Visual Complexity Regression

Estimates how visually complex the scene is:

- **Simple**: Clear sky, flat ground, one object
- **Moderate**: Varied terrain, several objects
- **Complex**: Dense structures, many entities, detailed environment

Example:

```text
Complexity Score: 78 / 100
```

---

## 3. Color Distribution Regression

Estimates the dominant colors in the image:

```text
Dominant Colors:
- Green:    45% (grass, leaves)
- Blue:     25% (sky, water)
- Brown:    15% (earth, wood)
- Gray:     10% (stone, concrete)
- White:     5% (clouds, snow)
```

This helps understand the visual characteristics of the generated art.

---

## 4. Minecraft Style Regression

For image-to-image mode, the model can predict how close the generated image matches Minecraft style:

```text
Style Score: 89 / 100 (Very High Similarity)

Style Features:
- Pixelation:    95%
- Color Palette: 88%
- Texture Style: 87%
- Block Patterns: 86%
- Lighting:      83%
```

---

## 5. Pose Estimation Regression

For character images, estimate pose:

```text
Pose Estimation:

- Arm 1: Up (35°)
- Arm 2: Down (0°)
- Head Angle: Slightly Left (10°)
- Body Lean: Neutral (0°)
```

---

## 6. Minecraft Game State Regression

Estimate game-related properties:

```text
Game State:
- Time of Day:         ~6:00 AM (Sunrise)
- Weather:             Clear
- Light Level:         High (8/15)
- Block Density:       Moderate
- Biome Type:          Plains / Forest
- mob Count:           2
- Structure Probability: Low
```

---

## Summary

| Regression Type | Prediction Type          |
| --------------- | ------------------------ |
| Composition     | Percentage values        |
| Complexity      | 0-100 score              |
| Color Dist.     | Color percentages        |
| Style Score     | Style similarity (0-100) |
| Pose Est.       | Angle values             |
| Game State      | Game-related metrics     |

Regression provides **continuous numerical insights** into the image content that classification alone cannot capture.
