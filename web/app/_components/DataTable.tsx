"use client";

import { ReactNode } from "react";

type DataTableProps = {
  title?: string;
  subtitle?: string;
  columns: Array<string | ReactNode>;
  rows?: Array<Array<ReactNode>>;
  loading?: boolean;
  emptyText?: string;
  actions?: ReactNode;
  footer?: ReactNode;
  skeletonRows?: number;
};

export default function DataTable({
  title,
  subtitle,
  columns,
  rows,
  loading = false,
  emptyText = "No data available.",
  actions,
  footer,
  skeletonRows = 4
}: DataTableProps) {
  const hasRows = rows && rows.length > 0;

  return (
    <div className="card table-card">
      {(title || subtitle || actions) && (
        <div className="panel-header">
          <div>
            {title && <h3>{title}</h3>}
            {subtitle && <p className="muted">{subtitle}</p>}
          </div>
          {actions && <div className="table-actions">{actions}</div>}
        </div>
      )}

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              {columns.map((column, index) => (
                <th key={typeof column === "string" ? column : `col-${index}`}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading &&
              Array.from({ length: skeletonRows }).map((_, rowIndex) => (
                <tr key={`skeleton-${rowIndex}`}>
                  {columns.map((column, colIndex) => (
                    <td key={`${column}-${colIndex}`}>
                      <div className="skeleton skeleton-line" />
                    </td>
                  ))}
                </tr>
              ))}

            {!loading &&
              hasRows &&
              rows?.map((row, rowIndex) => (
                <tr key={`row-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`cell-${rowIndex}-${cellIndex}`}>{cell}</td>
                  ))}
                </tr>
              ))}

            {!loading && !hasRows && (
              <tr>
                <td colSpan={columns.length} className="table-empty">
                  {emptyText}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {footer && <div className="table-footer">{footer}</div>}
    </div>
  );
}
