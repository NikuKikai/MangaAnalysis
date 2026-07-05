type SliderControlProps = {
  label: string;
  title: string;
  min: string;
  max: string;
  step: string;
  value: number;
  displayValue: string;
  onChange: (value: number) => void;
};

export function SliderControl({ label, title, min, max, step, value, displayValue, onChange }: SliderControlProps) {
  return (
    <label className="slider-row" title={title}>
      <span className="slider-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <span className="slider-value">{displayValue}</span>
    </label>
  );
}
