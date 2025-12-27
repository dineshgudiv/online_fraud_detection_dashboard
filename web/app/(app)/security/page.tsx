"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import DataTable from "@/app/_components/DataTable";
import StatCard from "@/app/_components/StatCard";
import CertificateViewer from "@/app/(app)/security/_components/CertificateViewer";
import { useAuth } from "@/app/_components/AuthProvider";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import {
  SecurityCertList,
  SecurityHsmProvider,
  SecurityVenafiStatus
} from "@/lib/security";
import { downloadFile } from "@/lib/utils";

export default function SecurityCenterOverview() {
  const { can } = useAuth();
  const [active, setActive] = useState<SecurityCertList | null>(null);
  const [expiring, setExpiring] = useState<SecurityCertList | null>(null);
  const [venafi, setVenafi] = useState<SecurityVenafiStatus | null>(null);
  const [hsm, setHsm] = useState<SecurityHsmProvider | null>(null);
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
        const [activeResp, expiringResp, venafiResp, hsmResp] =
          await Promise.all([
            apiFetch<SecurityCertList>("/security/pki/certs"),
            apiFetch<SecurityCertList>("/security/pki/certs/expiring?days=30"),
            apiFetch<SecurityVenafiStatus>("/security/venafi/status"),
            apiFetch<SecurityHsmProvider>("/security/hsm/provider")
          ]);
        if (!mounted) {
          return;
        }
        setActive(activeResp);
        setExpiring(expiringResp);
        setVenafi(venafiResp);
        setHsm(hsmResp);
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load security data");
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

  const activeRows = useMemo(() => {
    return (active?.items ?? []).map((cert) => [
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
  }, [active, can, handleDownload, openViewer, renderStatus]);

  const expiringRows = useMemo(() => {
    return (expiring?.items ?? []).map((cert) => [
      cert.common_name,
      cert.issuer,
      formatDateUTC(cert.not_after),
      renderStatus(cert.status),
      <div key={`expiring-actions-${cert.id}`} className="row-actions">
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

  const venafiValue = venafi
    ? venafi.connected
      ? `Connected${venafi.last_sync ? ` - ${formatDateUTC(venafi.last_sync)}` : ""}`
      : "Not connected"
    : "Loading...";

  const hsmValue = hsm
    ? hsm.configured
      ? `${hsm.provider ?? "Configured"} - keys ${hsm.key_count}`
      : "Not configured"
    : "Loading...";

  return (
    <div className="section">
      {error && <div className="error">{error}</div>}
      {actionError && <div className="error">{actionError}</div>}

      <div className="grid">
        <StatCard
          label="Certificates (active)"
          value={loading ? "Loading..." : String(active?.count ?? 0)}
          hint="Tracked certificates in inventory"
        />
        <StatCard
          label="Expiring (<= 30 days)"
          value={loading ? "Loading..." : String(expiring?.count ?? 0)}
          hint="Renewal risk window"
        />
        <StatCard
          label="Venafi sync"
          value={venafiValue}
          hint={venafi?.health ?? ""}
        />
        <StatCard
          label="HSM provider"
          value={hsmValue}
          hint={hsm?.health ?? ""}
        />
      </div>

      <DataTable
        title="Active certificates"
        subtitle="Inventory of currently tracked certificates."
        columns={["Common name", "Issuer", "Not after", "Status", "Actions"]}
        rows={activeRows}
        loading={loading}
        emptyText="No certificates found. Enable SECURITY_DEMO_MODE or connect PKI inventory."
      />

      <DataTable
        title="Expiring soon"
        subtitle="Certificates with upcoming renewal deadlines."
        columns={["Common name", "Issuer", "Not after", "Status", "Actions"]}
        rows={expiringRows}
        loading={loading}
        emptyText="No certificates expiring in the next 30 days."
      />

      <CertificateViewer
        open={viewerOpen}
        certId={selectedCertId}
        onClose={closeViewer}
      />
    </div>
  );
}
