"use client";

import { useEffect, useMemo, useState } from "react";

import DataTable from "@/app/_components/DataTable";
import StatCard from "@/app/_components/StatCard";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { SecurityCertList } from "@/lib/security";

export default function PKIPage() {
  const [certs, setCerts] = useState<SecurityCertList | null>(null);
  const [expiring, setExpiring] = useState<SecurityCertList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const [certsResp, expiringResp] = await Promise.all([
          apiFetch<SecurityCertList>("/security/pki/certs"),
          apiFetch<SecurityCertList>("/security/pki/certs/expiring?days=30")
        ]);
        if (!mounted) {
          return;
        }
        setCerts(certsResp);
        setExpiring(expiringResp);
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load PKI inventory");
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

  const certItems = certs?.items ?? [];
  const expiringSet = useMemo(
    () => new Set((expiring?.items ?? []).map((item) => item.id)),
    [expiring]
  );
  const issuerCount = useMemo(() => new Set(certItems.map((item) => item.issuer)).size, [certItems]);
  const revokedCount = useMemo(
    () => certItems.filter((item) => item.status !== "ACTIVE").length,
    [certItems]
  );
  const ocspStatus = loading ? "Loading..." : certs ? "Monitoring" : "Not configured";

  const filtered = useMemo(() => {
    let items = certItems;
    if (statusFilter === "expiring") {
      items = items.filter((item) => expiringSet.has(item.id));
    } else if (statusFilter === "active") {
      items = items.filter((item) => item.status === "ACTIVE");
    } else if (statusFilter === "revoked") {
      items = items.filter((item) => item.status !== "ACTIVE");
    }
    if (!query) {
      return items;
    }
    const needle = query.toLowerCase();
    return items.filter((item) => {
      return (
        item.common_name.toLowerCase().includes(needle) ||
        item.issuer.toLowerCase().includes(needle)
      );
    });
  }, [certItems, expiringSet, query, statusFilter]);

  const rows = useMemo(() => {
    return filtered.map((item) => [
      item.common_name,
      item.issuer,
      formatDateUTC(item.not_after),
      item.status
    ]);
  }, [filtered]);

  return (
    <div className="section">
      {error && <div className="error">{error}</div>}

      <div className="grid">
        <StatCard
          label="Trusted issuers"
          value={loading ? "Loading..." : String(issuerCount)}
          hint="Unique CAs currently in inventory"
        />
        <StatCard
          label="Revoked or invalid"
          value={loading ? "Loading..." : String(revokedCount)}
          hint="Certificates requiring review"
        />
        <StatCard label="OCSP and CRL health" value={ocspStatus} hint="Endpoint monitoring" />
      </div>

      <DataTable
        title="Certificate inventory"
        subtitle="CN or SAN, issuer, expiry, status."
        columns={["Common name", "Issuer", "Expires", "Status"]}
        rows={rows}
        loading={loading}
        emptyText="No certificates found. Enable SECURITY_DEMO_MODE or connect PKI inventory."
        actions={
          <div className="toolbar">
            <input
              placeholder="Search CN or issuer"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <select
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value)}
            >
              <option value="all">All status</option>
              <option value="active">Active</option>
              <option value="expiring">Expiring</option>
              <option value="revoked">Revoked</option>
            </select>
          </div>
        }
      />
    </div>
  );
}
