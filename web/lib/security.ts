export type SecurityCert = {
  id: string;
  common_name: string;
  issuer: string;
  not_before: string;
  not_after: string;
  serial: string;
  status: string;
  days_to_expiry?: number | null;
  trust_state?: string | null;
};

export type SecurityCertList = {
  items: SecurityCert[];
  count: number;
};

export type SecuritySan = {
  dns?: string[];
  ip?: string[];
};

export type SecurityCertChainNode = {
  label: string;
  type: string;
  current?: boolean;
};

export type SecurityCertDetail = SecurityCert & {
  signature_algorithm?: string | null;
  public_key_algorithm?: string | null;
  key_algorithm?: string | null;
  key_size?: number | null;
  san?: SecuritySan | null;
  key_usage?: string[] | null;
  enhanced_key_usage?: string[] | null;
  pem?: string | null;
  chain?: SecurityCertChainNode[] | null;
  trust_state?: string | null;
  trust_reason?: string | null;
};

export type SecurityVenafiStatus = {
  connected: boolean;
  last_sync: string | null;
  profile: string | null;
  health: string;
};

export type SecurityHsmProvider = {
  configured: boolean;
  provider: string | null;
  key_count: number;
  rotation_policy: string | null;
  health: string;
};
