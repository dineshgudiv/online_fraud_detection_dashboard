"use client";

import { useEffect, useMemo, useState } from "react";

import { useAuth } from "@/app/_components/AuthProvider";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { SecurityCertDetail, SecurityCertChainNode } from "@/lib/security";
import { copyToClipboard, downloadFile } from "@/lib/utils";

type ViewerTab = "General" | "Details" | "Certification Path";

type CertificateViewerProps = {
  open: boolean;
  certId: string | null;
  onClose: () => void;
};

const TABS: ViewerTab[] = ["General", "Details", "Certification Path"];

const skeletonLine = (width = "60%") => (
  <div className="skeleton skeleton-line" style={{ width }} />
);

const formatList = (values?: string[] | null) => {
  if (!values || values.length === 0) {
    return "n/a";
  }
  return values.join(", ");
};

const formatSan = (detail: SecurityCertDetail | null) => {
  const dns = detail?.san?.dns ?? [];
  const ip = detail?.san?.ip ?? [];
  const parts: string[] = [];
  if (dns.length) {
    parts.push(`DNS: ${dns.join(", ")}`);
  }
  if (ip.length) {
    parts.push(`IP: ${ip.join(", ")}`);
  }
  return parts.length ? parts.join(" | ") : "n/a";
};

const formatKeyUsage = (detail: SecurityCertDetail | null) => {
  const keyUsage = formatList(detail?.key_usage);
  const eku = formatList(detail?.enhanced_key_usage);
  if (keyUsage === "n/a" && eku === "n/a") {
    return "n/a";
  }
  if (keyUsage === "n/a") {
    return eku;
  }
  if (eku === "n/a") {
    return keyUsage;
  }
  return `${keyUsage} | ${eku}`;
};

const formatDaysUntil = (detail: SecurityCertDetail | null) => {
  if (detail?.days_to_expiry !== null && detail?.days_to_expiry !== undefined) {
    const days = detail.days_to_expiry;
    if (days < 0) {
      return `Expired ${Math.abs(days)} days ago`;
    }
    return `${days} days`;
  }
  if (!detail?.not_after) {
    return "n/a";
  }
  const expires = new Date(detail.not_after);
  if (Number.isNaN(expires.getTime())) {
    return "n/a";
  }
  const now = new Date();
  const diffMs = expires.getTime() - now.getTime();
  const days = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
  if (days < 0) {
    return `Expired ${Math.abs(days)} days ago`;
  }
  return `${days} days`;
};

const chainFallback = (detail: SecurityCertDetail | null): SecurityCertChainNode[] => {
  if (!detail?.common_name) {
    return [];
  }
  return [
    { label: "Root CA", type: "Root CA" },
    { label: "Intermediate CA", type: "Intermediate CA" },
    { label: detail.common_name, type: "Leaf Certificate", current: true },
  ];
};

export default function CertificateViewer({ open, certId, onClose }: CertificateViewerProps) {
  const { can } = useAuth();
  const [activeTab, setActiveTab] = useState<ViewerTab>("General");
  const [detail, setDetail] = useState<SecurityCertDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open || !certId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    setError(null);
    setCopyMessage(null);
    setActiveTab("General");
    apiFetch<SecurityCertDetail>(`/security/pki/certs/${certId}`)
      .then((resp) => {
        setDetail(resp);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load certificate");
      })
      .finally(() => {
        setLoading(false);
      });
  }, [open, certId]);

  const status = detail?.status ?? "n/a";
  const statusClass =
    status === "ACTIVE" || status === "EXPIRED" || status === "REVOKED" ? status.toLowerCase() : "unknown";
  const trustState = detail?.trust_state ?? (status === "ACTIVE" ? "VALID" : "UNKNOWN");
  const isTrusted = trustState === "VALID";

  const chainNodes = useMemo(() => {
    if (detail?.chain && detail.chain.length > 0) {
      return detail.chain;
    }
    return chainFallback(detail);
  }, [detail]);

  const handleCopyPem = async () => {
    if (!detail?.pem) {
      setCopyMessage("PEM not available");
      return;
    }
    try {
      await copyToClipboard(detail.pem);
      setCopyMessage("PEM copied");
    } catch (err) {
      setCopyMessage(err instanceof Error ? err.message : "Copy failed");
    } finally {
      setTimeout(() => setCopyMessage(null), 2000);
    }
  };

  const handleDownload = async () => {
    if (!can("security:cert:download")) {
      setError("Not authorized to download certificates");
      return;
    }
    if (!certId) {
      return;
    }
    try {
      await downloadFile(`/api/proxy/security/pki/certs/${certId}/download`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    }
  };

  if (!open) {
    return null;
  }

  return (
    <div className="cert-viewer-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="cert-viewer" onClick={(event) => event.stopPropagation()}>
        <div className="cert-viewer-header">
          <div className="cert-title">
            <h3>{loading ? "Loading certificate..." : detail?.common_name ?? "Certificate viewer"}</h3>
            <p className="muted">
              {loading ? "Fetching certificate metadata" : detail?.issuer ?? "Certificate authority"}
            </p>
          </div>
          <div className={`status-badge ${statusClass}`}>{status}</div>
        </div>

        <div className="cert-viewer-body">
          {error && <div className="error">{error}</div>}

          <div className="tabs cert-tabs">
            {TABS.map((tab) => (
              <button
                type="button"
                key={tab}
                className={`tab ${tab === activeTab ? "active" : ""}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
              </button>
            ))}
          </div>

          {activeTab === "General" && (
            <div className="cert-tab-panel">
              <div className="kv-grid">
                <div className="kv-item">
                  <div className="kv-label">Issued To</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.common_name ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Issued By</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.issuer ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Valid From</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : formatDateUTC(detail?.not_before)}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Valid To</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : formatDateUTC(detail?.not_after)}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Status</div>
                  <div className="kv-value">
                    {loading ? skeletonLine("40%") : <span className={`status-badge ${statusClass}`}>{status}</span>}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Days Until Expiry</div>
                  <div className="kv-value">
                    {loading ? skeletonLine("50%") : formatDaysUntil(detail)}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Enhanced Key Usage</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : formatList(detail?.enhanced_key_usage)}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Trust Status</div>
                  <div className="kv-value">
                    {loading ? (
                      skeletonLine("45%")
                    ) : (
                      <span className={`trust-indicator ${isTrusted ? "trusted" : "untrusted"}`}>
                        {isTrusted ? "Trusted" : trustState.split("_").join(" ")}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              {!loading && trustState !== "VALID" && (
                <div className="error-banner">
                  Trust warning: {detail?.trust_reason ?? "Certificate failed validation checks."}
                </div>
              )}
            </div>
          )}

          {activeTab === "Details" && (
            <div className="cert-tab-panel">
              <div className="kv-grid">
                <div className="kv-item">
                  <div className="kv-label">Common Name (CN)</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.common_name ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Issuer</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.issuer ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Serial Number</div>
                  <div className="kv-value mono">
                    {loading ? skeletonLine("55%") : detail?.serial ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Signature Algorithm</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.signature_algorithm ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Public Key Algorithm</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.public_key_algorithm ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Key Algorithm</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.key_algorithm ?? "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Key Size</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : detail?.key_size ? `${detail.key_size} bits` : "n/a"}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">SAN</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : formatSan(detail)}
                  </div>
                </div>
                <div className="kv-item">
                  <div className="kv-label">Key Usage / EKU</div>
                  <div className="kv-value">
                    {loading ? skeletonLine() : formatKeyUsage(detail)}
                  </div>
                </div>
              </div>

              <div className="pem-details">
                <div className="pem-header">
                  <h4>Certificate PEM</h4>
                  <button
                    type="button"
                    className="button secondary compact"
                    onClick={handleCopyPem}
                    disabled={!detail?.pem}
                  >
                    Copy PEM
                  </button>
                </div>
                {copyMessage && <div className="muted">{copyMessage}</div>}
                <details>
                  <summary>View PEM</summary>
                  <pre className="pem-block mono">
                    {loading ? "Loading PEM..." : detail?.pem ?? "n/a"}
                  </pre>
                </details>
              </div>
            </div>
          )}

          {activeTab === "Certification Path" && (
            <div className="cert-tab-panel">
              {!loading && trustState !== "VALID" && (
                <div className="error-banner">
                  Trust warning: {detail?.trust_reason ?? "Certificate chain issues detected."}
                </div>
              )}
              <div className="cert-chain">
                {chainNodes.length === 0 && (
                  <div className="empty">No certification path available.</div>
                )}
                {chainNodes.map((node, index) => (
                  <div
                    key={`${node.label}-${index}`}
                    className={`cert-chain-item ${node.current ? "current" : ""}`}
                  >
                    <div className="cert-chain-title">{node.label}</div>
                    <div className="muted">{node.type}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="cert-viewer-footer">
          {can("security:cert:download") && (
            <button type="button" className="button secondary compact" onClick={handleDownload}>
              Download .crt
            </button>
          )}
          <button type="button" className="button secondary compact" onClick={handleCopyPem}>
            Copy PEM
          </button>
          <button type="button" className="button compact" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
