"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import DataTable from "@/app/_components/DataTable";
import StatCard from "@/app/_components/StatCard";
import CertificateViewer from "@/app/(app)/security/_components/CertificateViewer";
import { useAuth } from "@/app/_components/AuthProvider";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { SecurityCertList, SecurityVenafiStatus } from "@/lib/security";
import { downloadFile } from "@/lib/utils";

export default function CertificatesPage() {
  const { can } = useAuth();
  const [certs, setCerts] = useState<SecurityCertList | null>(null);
  const [expiring, setExpiring] = useState<SecurityCertList | null>(null);
  const [venafi, setVenafi] = useState<SecurityVenafiStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [viewerOpen, setViewerOpen] = useState(false);
  const [selectedCertId, setSelectedCertId] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [certsResp, expiringResp, venafiResp] = await Promise.all([
          apiFetch<SecurityCertList>("/security/pki/certs"),
          apiFetch<SecurityCertList>("/security/pki/certs/expiring?days=30"),
          apiFetch<SecurityVenafiStatus>("/security/venafi/status")
        ]);
        if (!mounted) {
          return;
        }
        setCerts(certsResp);
        setExpiring(expiringResp);
        setVenafi(venafiResp);
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load Venafi data");
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    load();
    return () => {
      mounted = false;
    };
  }, []);

  const openViewer = useCallback((certId: string) => {
    setSelectedCertId(certId);
    setViewerOpen(true);
  }, []);

  const closeViewer = useCallback(() => {
    setViewerOpen(false);
  }, []);

  const handleDownload = useCallback(async (certId: string) => {
    setActionError(null);
    try {
      await downloadFile(`/api/proxy/security/pki/certs/${certId}/download`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Download failed");
    }
  }, []);

  const renderStatus = useCallback((status: string) => {
    const className =
      status === "ACTIVE" || status === "EXPIRED" || status === "REVOKED"
        ? status.toLowerCase()
        : "unknown";
    return <span className={`status-badge ${className}`}>{status}</span>;
  }, []);

  const total = certs?.count ?? 0;
  const managed = venafi?.connected ? total : 0;
  const unmanaged = Math.max(total - managed, 0);
  const renewalDue = expiring?.count ?? 0;

  const venafiStatusLabel = venafi
    ? venafi.connected
      ? "Venafi: connected"
      : "Venafi: not connected"
    : "Venafi: loading";

  const expiringRows = useMemo(() => {
    return (expiring?.items ?? []).map((cert) => [
      cert.common_name,
      cert.issuer,
      formatDateUTC(cert.not_after),
      renderStatus(cert.status),
      <div key={`actions-${cert.id}`} className="row-actions">
        <button
          type="button"
          className="button secondary compact"
          onClick={() => openViewer(cert.id)}
        >
          View
        </button>
        {can("security:cert:download") && (
          <button
            type="button"
            className="button compact"
            onClick={() => handleDownload(cert.id)}
          >
            Download
          </button>
        )}
      </div>
    ]);
  }, [expiring, can, handleDownload, openViewer, renderStatus]);

  return (
    <div className="section">
      {error && <div className="error">{error}</div>}
      {actionError && <div className="error">{actionError}</div>}

      <div className="card">
        <div className="panel-header">
          <div>
            <h3>Certificates (Venafi CSM)</h3>
            <p className="muted">
              Managed vs unmanaged coverage, renewal queue, and policy
              compliance.
            </p>
            {venafi && (
              <p className="muted">
                Last sync: {venafi.last_sync ? formatDateUTC(venafi.last_sync) : "No sync yet"} -{" "}
                Health: {venafi.health}
              </p>
            )}
          </div>
          <div className="status-pill">{venafiStatusLabel}</div>
        </div>
      </div>

      <div className="grid">
        <StatCard
          label="Managed by Venafi"
          value={loading ? "Loading..." : String(managed)}
          hint="Certificates under policy control"
        />
        <StatCard
          label="Unmanaged"
          value={loading ? "Loading..." : String(unmanaged)}
          hint="Inventory outside Venafi coverage"
        />
        <StatCard
          label="Renewals due"
          value={loading ? "Loading..." : String(renewalDue)}
          hint="Expiring within 30 days"
        />
      </div>

      <DataTable
        title="Renewal queue"
        subtitle="Certificates expiring soon and awaiting renewal."
        columns={["Common name", "Issuer", "Not after", "Status", "Actions"]}
        rows={expiringRows}
        loading={loading}
        emptyText="No upcoming renewals in the next 30 days."
      />

      <CertificateViewer
        open={viewerOpen}
        certId={selectedCertId}
        onClose={closeViewer}
      />
    </div>
  );
}
