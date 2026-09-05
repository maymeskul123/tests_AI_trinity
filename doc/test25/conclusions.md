# Выводы по test25

- **QuaternaryThreshold** даёт лучший компромисс: false_confidence ≈ 0.10 (против 0.38 у Quaternary), reward ≈ –10.9 (против –38.6).
- abstain всё ещё высок (0.88), но значительно лучше, чем у обычного Quaternary.
- **Вывод:** Динамический порог – эффективное улучшение Quaternary.
