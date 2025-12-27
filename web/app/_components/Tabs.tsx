"use client";

type Tab = {
  label: string;
  value: string;
  count?: number;
};

type TabsProps = {
  tabs: Tab[];
  active: string;
  onChange: (value: string) => void;
};

export default function Tabs({ tabs, active, onChange }: TabsProps) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => {
        const isActive = tab.value === active;
        return (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={`tab${isActive ? " active" : ""}`}
            onClick={() => onChange(tab.value)}
          >
            <span>{tab.label}</span>
            {typeof tab.count === "number" && (
              <span className="tab-count">{tab.count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
