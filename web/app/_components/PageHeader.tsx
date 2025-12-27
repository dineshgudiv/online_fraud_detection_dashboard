"use client";

import Link from "next/link";
import { ReactNode } from "react";

export type Breadcrumb = {
  label: string;
  href?: string;
};

type PageHeaderProps = {
  title: string;
  subtitle?: string;
  breadcrumbs?: Breadcrumb[];
  actions?: ReactNode;
};

export default function PageHeader({
  title,
  subtitle,
  breadcrumbs,
  actions
}: PageHeaderProps) {
  return (
    <div className="page-header">
      <div className="page-header-text">
        {breadcrumbs && breadcrumbs.length > 0 && (
          <nav className="breadcrumbs" aria-label="Breadcrumb">
            {breadcrumbs.map((crumb, index) => (
              <span key={`${crumb.label}-${index}`} className="breadcrumb-item">
                {crumb.href ? (
                  <Link href={crumb.href}>{crumb.label}</Link>
                ) : (
                  <span>{crumb.label}</span>
                )}
                {index < breadcrumbs.length - 1 && (
                  <span className="breadcrumb-sep">/</span>
                )}
              </span>
            ))}
          </nav>
        )}
        <h3>{title}</h3>
        {subtitle && <p className="muted">{subtitle}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}
